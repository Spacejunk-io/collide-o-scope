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
    /// A deterministic synthetic field computed by the closed B2 vocabulary.
    /// It needs no codec, no luma observation history, and no fallback: the
    /// same authored kind, scale, rate, and program time always produce the
    /// same field, live and offline.
    Procedural(ProceduralFieldKind),
}

impl MotionFieldSource {
    /// The authored procedural kind, if this source selects one.
    pub const fn procedural_kind(self) -> Option<ProceduralFieldKind> {
        match self {
            Self::Procedural(kind) => Some(kind),
            _ => None,
        }
    }
}

/// The closed B2 procedural field vocabulary.
///
/// Curl, Radial, Spiral, and Weave are pure functions of UV, program time, and
/// the two authored scalars; their gates are fully open. Contour and Chroma
/// additionally read the recipient's own image — alpha-covered, so hostile
/// hidden RGB at zero coverage steers nothing — and report an honest gradient/
/// saturation confidence instead of pretending flat content moves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProceduralFieldKind {
    #[default]
    Curl,
    Radial,
    Spiral,
    Contour,
    Chroma,
    Weave,
}

impl ProceduralFieldKind {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Curl => 0,
            Self::Radial => 1,
            Self::Spiral => 2,
            Self::Contour => 3,
            Self::Chroma => 4,
            Self::Weave => 5,
        }
    }

    /// Every kind in the closed vocabulary, in code order.
    pub const ALL: [Self; 6] = [
        Self::Curl,
        Self::Radial,
        Self::Spiral,
        Self::Contour,
        Self::Chroma,
        Self::Weave,
    ];

    /// True when this kind observes the recipient's image. Contour steers
    /// along luma isolines; Chroma steers by the YIQ chroma pair. The other
    /// four are pure functions of UV and time and bind no image at all.
    pub const fn reads_image(self) -> bool {
        matches!(self, Self::Contour | Self::Chroma)
    }

    /// The stable `field_source` wire/sidecar token for this kind. Every
    /// stringify site answers from this one table so the wire, the snapshot,
    /// and the sidecar can never disagree about a kind's name.
    pub const fn source_key(self) -> &'static str {
        match self {
            Self::Curl => "procedural_curl",
            Self::Radial => "procedural_radial",
            Self::Spiral => "procedural_spiral",
            Self::Contour => "procedural_contour",
            Self::Chroma => "procedural_chroma",
            Self::Weave => "procedural_weave",
        }
    }
}

impl fmt::Display for ProceduralFieldKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Curl => write!(f, "curl"),
            Self::Radial => write!(f, "radial"),
            Self::Spiral => write!(f, "spiral"),
            Self::Contour => write!(f, "contour"),
            Self::Chroma => write!(f, "chroma"),
            Self::Weave => write!(f, "weave"),
        }
    }
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

/// Authored scalars for the procedural field family.
///
/// Both values are scope state like the shutter's: they persist and modulate
/// whether or not the current `field_source` is procedural, so switching kinds
/// never erases what the operator dialed in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProceduralFieldParams {
    /// Spatial density in `[0, 1]`; the field law maps it to 1–16 cycles
    /// across the frame.
    pub scale: f32,
    /// Signed animation rate in turns per second, clamped to `[-2, 2]`.
    pub rate: f32,
}

impl Default for ProceduralFieldParams {
    fn default() -> Self {
        Self {
            scale: 0.5,
            rate: 0.25,
        }
    }
}

impl ProceduralFieldParams {
    pub fn sanitized(self) -> Self {
        Self {
            scale: unit(self.scale, 0.5),
            rate: finite_or(self.rate, 0.25).clamp(-2.0, 2.0),
        }
    }
}

/// Authored B2 flow-shaping controls.
///
/// These shape the field the advection pass *applies* — after sampling and
/// gating, before the trajectory offset — so they act on every field kind:
/// codec, lattice, procedural, or the derived collided field. All three
/// amounts at zero are the exact prior path: the shader takes no extra
/// texture operation and adds nothing to the velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowShapingParams {
    /// Radial growth by local field magnitude. Zero is exact off.
    pub stretch: f32,
    /// Push away from the carrier's luma gradient. Zero is exact off.
    pub edge_repel: f32,
    /// Probability per cell per event tick that a hashed garbage vector
    /// shoves the cell. Zero is exact off.
    pub vector_trash: f32,
    /// Trash cell edge in output pixels, clamped to `[2, 256]`.
    pub trash_block_size: f32,
}

impl Default for FlowShapingParams {
    fn default() -> Self {
        Self {
            stretch: 0.0,
            edge_repel: 0.0,
            vector_trash: 0.0,
            trash_block_size: 16.0,
        }
    }
}

impl FlowShapingParams {
    pub fn sanitized(self) -> Self {
        Self {
            stretch: unit(self.stretch, 0.0),
            edge_repel: unit(self.edge_repel, 0.0),
            vector_trash: unit(self.vector_trash, 0.0),
            trash_block_size: finite_or(self.trash_block_size, 16.0).clamp(2.0, 256.0),
        }
    }

    /// True when shaping is the exact prior path.
    pub fn is_exact_zero(self) -> bool {
        let sanitized = self.sanitized();
        sanitized.stretch == 0.0 && sanitized.edge_repel == 0.0 && sanitized.vector_trash == 0.0
    }
}

/// Complete M4 authored motion contract for one master or layer scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionParams {
    pub algorithm_version: u16,
    pub field_source: MotionFieldSource,
    pub lattice_quality: MotionLatticeQuality,
    /// B2 procedural field scalars. Inert unless `field_source` is
    /// `Procedural(_)`.
    pub procedural: ProceduralFieldParams,
    /// B2 flow shaping applied where the field is consumed. All-zero is the
    /// exact prior advection path.
    pub shaping: FlowShapingParams,
    pub transplant: FaradayParams,
    pub shutter: CurvedShutterParams,
    /// Field Collider v1. Disabled is exact M4: the block delegates before any
    /// admission or allocation and the single-donor transplant recipe below
    /// runs untouched. Enabling it *parks* that recipe — the authored donor,
    /// amount, carrier, confidence, refresh, decay, and occlusion are all
    /// retained verbatim — and substitutes the derived collided field as the
    /// thing the carrier is advected from. Disabling resumes the parked recipe
    /// exactly, because nothing about it was ever erased.
    pub collider: FieldColliderParams,
}

impl Default for MotionParams {
    fn default() -> Self {
        Self {
            algorithm_version: MOTION_ALGORITHM_VERSION,
            field_source: MotionFieldSource::Auto,
            lattice_quality: MotionLatticeQuality::Live,
            procedural: ProceduralFieldParams::default(),
            shaping: FlowShapingParams::default(),
            transplant: FaradayParams::default(),
            shutter: CurvedShutterParams::default(),
            collider: FieldColliderParams::default(),
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
            procedural: self.procedural.sanitized(),
            shaping: self.shaping.sanitized(),
            transplant: self.transplant.sanitized(),
            shutter: self.shutter.sanitized(),
            collider: self.collider.sanitized(),
        }
    }

    pub fn is_exact_zero(self) -> bool {
        self.transplant.amount == 0.0 && self.shutter.angle_degrees == 0.0
    }

    /// The admitted Field Collider for this scope, if any.
    ///
    /// A collider derives the field the Faraday transplant advects from, so it
    /// is meaningless without an active transplant: with `amount == 0.0` there
    /// is no carrier to advect and no pass to feed. It is likewise refused at
    /// master scope, where a transplant is already refused. Both conditions
    /// are answered here, so every call site asks the identical question.
    pub fn collider_admission(self, is_master: bool) -> FieldColliderAdmission {
        let collider = self.collider.sanitized();
        // Authored inertness is reported before any environmental fault, so a
        // disabled block never accuses the scope of a problem it does not have.
        if !collider.enabled {
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::Disabled,
            };
        }
        if is_master {
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::MasterRecipient,
            };
        }
        if self.transplant.sanitized().amount <= 0.0 {
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::NoActiveTransplant,
            };
        }
        collider.admission()
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
    /// A B2 procedural field. It carries its kind because Contour and Chroma
    /// bind the recipient's image while the pure kinds bind nothing, so the
    /// kind is genuinely part of the prepared topology, not presentation.
    Procedural(ProceduralFieldKind),
}

impl MotionFieldOrigin {
    /// Append-only topology-signature code. `None`, `CodecVectors`,
    /// `Lattice`, and `LatticeFallback` keep their original 0–3; the six
    /// procedural kinds occupy 4–9 by their own permanent codes.
    pub const fn signature_code(self) -> u64 {
        match self {
            Self::None => 0,
            Self::CodecVectors => 1,
            Self::Lattice => 2,
            Self::LatticeFallback => 3,
            Self::Procedural(kind) => 4 + kind.code() as u64,
        }
    }

    /// The procedural kind this origin renders, if any.
    pub const fn procedural_kind(self) -> Option<ProceduralFieldKind> {
        match self {
            Self::Procedural(kind) => Some(kind),
            _ => None,
        }
    }
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
        // A procedural field is deterministic synthesis: it holds at master
        // and layer scope alike, needs no codec, and never falls back.
        MotionFieldSource::Procedural(kind) => MotionSourceDecision {
            origin: MotionFieldOrigin::Procedural(kind),
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
    /// A resolved Study ABI 1.1 program consumes this scope's primitive
    /// motion input. Like donor and Garden routing, this admits the field even
    /// when ordinary Motion effects are exactly zero.
    pub required_as_study_input: bool,
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
            let field_required = request.required_as_donor
                || request.required_as_garden_signal
                || request.required_as_study_input
                || shutter_active;
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
    TooManyColliders { count: u32, limit: u32 },
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
            Self::TooManyColliders { count, limit } => write!(
                f,
                "motion plan requests {count} field colliders; limit is {limit}"
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

// ---------------------------------------------------------------------------
// Field Collider v1
// ---------------------------------------------------------------------------
//
// The Field Collider is a Motion-subsystem block, not a Collision Rack node: it
// takes no `NodeKindTag` code, occupies no rack segment, and never appears in
// the image dependency graph. It consumes two admitted *primitive* motion
// fields, recombines their recipient-local vectors under a closed mode law, and
// publishes one derived vector/gate field. The existing Faraday transplant then
// advects its carrier from that derived field instead of from a single donor.
//
// Everything here is pure: no `wgpu`, no clock, no filesystem, no UI. The CPU
// laws below are the independent reference the GPU pass is measured against,
// not a description of the shader.

/// Persisted provenance for the first Field Collider algorithm family.
pub const FIELD_COLLIDER_ALGORITHM_VERSION: u16 = 1;

/// M4 admitted one expensive transplant; v1 likewise admits one collider and
/// therefore one derived field on top of that single carrier.
pub const MOTION_MAX_ACTIVE_COLLIDERS: u32 = 1;

// The collider's own per-cell working set. Both primitive inputs and the sole
// carrier stay separately and honestly accounted through the M4 ledger; only
// these three surfaces are new.
const DERIVED_VECTOR_BYTES_PER_CELL: u64 = 8; // two RG16Float parities
const DERIVED_GATE_BYTES_PER_CELL: u64 = 4; // two RG8Unorm parities
const TRANSIENT_PAIR_BYTES_PER_CELL: u64 = 8; // one RGBA16Float mapped pair
/// The complete published collider-specific delta, in bytes per grid cell.
pub const FIELD_COLLIDER_BYTES_PER_CELL: u64 =
    DERIVED_VECTOR_BYTES_PER_CELL + DERIVED_GATE_BYTES_PER_CELL + TRANSIENT_PAIR_BYTES_PER_CELL;

const _: () = assert!(FIELD_COLLIDER_BYTES_PER_CELL == 20);

/// Which of the two fixed collider inputs a value or diagnostic addresses.
///
/// Slot identity is authored topology, exactly as it is for a Symmetry Field's
/// image slots: an unarmed or missing input A can never slide input B's donor
/// down into its place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldColliderInput {
    A,
    B,
}

impl FieldColliderInput {
    /// Permanent append-only wire/persistence index. Never renumber.
    pub const fn index(self) -> u8 {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::A),
            1 => Some(Self::B),
            _ => None,
        }
    }

    pub const ALL: [Self; 2] = [Self::A, Self::B];
}

impl fmt::Display for FieldColliderInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
        }
    }
}

/// The closed v1 recombination vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldColliderMode {
    #[default]
    Sum,
    Difference,
    Curl,
    Projection,
    CollisionBoundary,
}

impl FieldColliderMode {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Sum => 0,
            Self::Difference => 1,
            Self::Curl => 2,
            Self::Projection => 3,
            Self::CollisionBoundary => 4,
        }
    }

    /// Every mode in the closed vocabulary, in code order.
    pub const ALL: [Self; 5] = [
        Self::Sum,
        Self::Difference,
        Self::Curl,
        Self::Projection,
        Self::CollisionBoundary,
    ];
}

/// Boundary law for a motion-field lookup outside its source extent.
///
/// The variants are declared in **shader-code order**, which is the frozen
/// `Transparent = 0, Mirror = 1, Wrap = 2, Hold = 3` numbering already carried
/// by [`crate::visual_rack::DisplaceBoundary`] and
/// [`crate::symmetry::SymmetryBoundary`]. Section 5 of the enrichment plan
/// lists the four names in the order "transparent, hold, mirror, wrap"; that
/// listing is prose enumerating the vocabulary, not a code assignment, and
/// minting a fourth incompatible boundary table so that motion disagreed with
/// the two image boundaries about the numeric meaning of `1` would be a
/// persistence and shader hazard for no authored benefit. Motion therefore
/// deliberately does **not** differ: one boundary numbering serves the whole
/// program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum MotionBoundaryMode {
    /// Inclusive `[0, 1]` acceptance. The only law that removes a sample.
    #[default]
    Transparent,
    /// Period-two triangular reflection.
    Mirror,
    /// `x - floor(x)`.
    Wrap,
    /// Clamp to the closed unit interval.
    Hold,
}

impl MotionBoundaryMode {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Transparent => 0,
            Self::Mirror => 1,
            Self::Wrap => 2,
            Self::Hold => 3,
        }
    }

    /// Every boundary in the closed vocabulary, in code order.
    pub const ALL: [Self; 4] = [Self::Transparent, Self::Mirror, Self::Wrap, Self::Hold];

    /// Resolve one lookup coordinate, or `None` when this law removes it.
    ///
    /// A non-finite coordinate is removed by every law, including the three
    /// that otherwise always produce a sample: `clamp`, `fract`, and the
    /// triangular map are all meaningless on NaN, and inventing a coordinate
    /// there would fabricate a reading the field never took.
    pub fn resolve(self, uv: [f32; 2]) -> Option<[f32; 2]> {
        if !uv[0].is_finite() || !uv[1].is_finite() {
            return None;
        }
        match self {
            Self::Transparent => {
                (uv[0] >= 0.0 && uv[0] <= 1.0 && uv[1] >= 0.0 && uv[1] <= 1.0).then_some(uv)
            }
            Self::Hold => Some([uv[0].clamp(0.0, 1.0), uv[1].clamp(0.0, 1.0)]),
            Self::Wrap => Some([wrap_unit(uv[0]), wrap_unit(uv[1])]),
            Self::Mirror => Some([mirror_unit(uv[0]), mirror_unit(uv[1])]),
        }
    }
}

fn wrap_unit(value: f32) -> f32 {
    // `value - floor(value)` on a large negative magnitude can round to exactly
    // 1.0; the closed unit interval is still the honest answer, so clamp rather
    // than letting a lookup escape by one ulp.
    (value - value.floor()).clamp(0.0, 1.0)
}

fn mirror_unit(value: f32) -> f32 {
    let half = value / 2.0;
    let period = (half - half.floor()) * 2.0;
    let folded = if period > 1.0 { 2.0 - period } else { period };
    folded.clamp(0.0, 1.0)
}

/// Clamp one derived velocity component into the canonical Motion range.
///
/// This is exactly the interval [`pack_velocity`] encodes and
/// [`unpack_velocity`] recovers, so no mode can emit a velocity the frozen M4
/// field contract cannot represent. It clamps without quantizing: the GPU
/// derived field is `Rg16Float`, so applying the 16-bit *lattice* here would
/// make the CPU reference disagree with the shader by construction.
pub fn clamp_motion_velocity(value: f32) -> f32 {
    finite_or(value, 0.0).clamp(-MOTION_MAX_UV_PER_SECOND, MOTION_MAX_UV_PER_SECOND)
}

/// Bounded authored Field Collider controls.
///
/// Version 1 adds no collider-only continuous control: the shared Faraday
/// `amount`, carrier, confidence threshold/softness, refresh, decay, and
/// occlusion remain the one carrier/advection law. Dice and modulation
/// therefore preserve this block exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldColliderParams {
    pub algorithm_version: u16,
    /// `false` is exact M4: the block delegates before any admission or
    /// allocation, and the single-donor transplant recipe resumes untouched.
    pub enabled: bool,
    pub mode: FieldColliderMode,
    pub boundary: MotionBoundaryMode,
    pub input_a: MotionDonor,
    pub input_b: MotionDonor,
}

impl Default for FieldColliderParams {
    fn default() -> Self {
        Self {
            algorithm_version: FIELD_COLLIDER_ALGORITHM_VERSION,
            enabled: false,
            mode: FieldColliderMode::Sum,
            boundary: MotionBoundaryMode::Transparent,
            input_a: MotionDonor::None,
            input_b: MotionDonor::None,
        }
    }
}

impl FieldColliderParams {
    pub fn sanitized(self) -> Self {
        Self {
            algorithm_version: FIELD_COLLIDER_ALGORITHM_VERSION,
            ..self
        }
    }

    /// The authored donor occupying one fixed input slot.
    pub const fn input(self, input: FieldColliderInput) -> MotionDonor {
        match input {
            FieldColliderInput::A => self.input_a,
            FieldColliderInput::B => self.input_b,
        }
    }

    pub const fn input_mut(&mut self, input: FieldColliderInput) -> &mut MotionDonor {
        match input {
            FieldColliderInput::A => &mut self.input_a,
            FieldColliderInput::B => &mut self.input_b,
        }
    }

    /// True when this block is exactly the frozen M4 recipe.
    ///
    /// This is the *delegation* predicate, and it is deliberately narrower than
    /// [`Self::admission`]: a disabled collider is inert authored state that
    /// carries no fault, whereas an enabled collider with a bad input also
    /// delegates but must stay visible.
    pub const fn is_exact_m4(self) -> bool {
        !self.enabled
    }

    /// Resolve the complete admission decision for this authored block.
    ///
    /// This single function is the whole admission law. The planner-collect,
    /// executor-encode, and dependency-walk sites all call *this* — the S1–S4
    /// three-site discipline is satisfied by construction rather than by three
    /// hand-copied predicates that can drift apart.
    pub fn admission(self) -> FieldColliderAdmission {
        if !self.enabled {
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::Disabled,
            };
        }
        let mut resolved = [None, None];
        for input in FieldColliderInput::ALL {
            match self.input(input) {
                MotionDonor::Selected { layer_id, .. } => {
                    resolved[usize::from(input.index())] = Some(layer_id);
                }
                MotionDonor::Missing { .. } => {
                    return FieldColliderAdmission::Delegated {
                        diagnostic: FieldColliderDiagnostic::InputMissing { input },
                    };
                }
                MotionDonor::None => {
                    return FieldColliderAdmission::Delegated {
                        diagnostic: FieldColliderDiagnostic::InputUnselected { input },
                    };
                }
            }
        }
        let [Some(a), Some(b)] = resolved else {
            // Unreachable: both slots were just proven `Selected`.
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::InputUnselected {
                    input: FieldColliderInput::A,
                },
            };
        };
        // A may equal the recipient and B may equal the recipient, but A and B
        // may never alias each other: colliding a field with itself is not a
        // second observation, and every mode would degenerate.
        if a == b {
            return FieldColliderAdmission::Delegated {
                diagnostic: FieldColliderDiagnostic::AliasedInputs,
            };
        }
        FieldColliderAdmission::Admitted {
            input_a: a,
            input_b: b,
        }
    }

    /// True exactly when the collider owns the recipient's motion field.
    pub fn is_admitted(self) -> bool {
        matches!(self.admission(), FieldColliderAdmission::Admitted { .. })
    }
}

/// The outcome of [`FieldColliderParams::admission`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldColliderAdmission {
    /// The block runs and owns the recipient's derived motion field.
    Admitted {
        input_a: StableLayerId,
        input_b: StableLayerId,
    },
    /// The block delegates to exact M4 before any admission or allocation.
    /// `Disabled` is authored inertness; every other diagnostic is a fault the
    /// operator must be able to see.
    Delegated { diagnostic: FieldColliderDiagnostic },
}

impl FieldColliderAdmission {
    pub const fn diagnostic(self) -> FieldColliderDiagnostic {
        match self {
            Self::Admitted { .. } => FieldColliderDiagnostic::None,
            Self::Delegated { diagnostic } => diagnostic,
        }
    }
}

/// Typed, telemetry-safe collider faults. These name authored identity only:
/// no host path, no filesystem metadata, and no pixel content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FieldColliderDiagnostic {
    #[default]
    None,
    /// Authored `enabled = false`. Not a fault: the exact M4 recipe is live.
    Disabled,
    /// The slot's saved donor did not survive reorder, removal, or replacement.
    /// A tombstone never rebinds; it stays visible until reauthored.
    InputMissing { input: FieldColliderInput },
    /// The slot names no donor at all.
    InputUnselected { input: FieldColliderInput },
    /// Both slots resolved to the same layer.
    AliasedInputs,
    /// The recipient is the master scope, which owns no Faraday carrier.
    MasterRecipient,
    /// The block is enabled, but its recipient has no active transplant, so
    /// there is no carrier to advect and nothing for a derived field to feed.
    NoActiveTransplant,
    /// The slot resolved, but its scope was not admitted a primitive field.
    InputFieldUnavailable { input: FieldColliderInput },
    /// The donor-local to recipient-local affine was non-finite or singular.
    SingularTransform { input: FieldColliderInput },
}

impl FieldColliderDiagnostic {
    /// True when this diagnostic describes a fault the operator should see.
    /// `None` and `Disabled` are ordinary states, not faults.
    pub const fn is_fault(self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

impl fmt::Display for FieldColliderDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "field collider admitted"),
            Self::Disabled => write!(f, "field collider disabled; exact M4 transplant is live"),
            Self::InputMissing { input } => write!(
                f,
                "field collider input {input} names a layer that is no longer present"
            ),
            Self::InputUnselected { input } => {
                write!(f, "field collider input {input} names no donor")
            }
            Self::AliasedInputs => write!(f, "field collider inputs A and B name the same layer"),
            Self::MasterRecipient => write!(f, "field collider requires a layer recipient"),
            Self::NoActiveTransplant => write!(
                f,
                "field collider requires an active Faraday transplant to advect"
            ),
            Self::InputFieldUnavailable { input } => write!(
                f,
                "field collider input {input} was not admitted a primitive motion field"
            ),
            Self::SingularTransform { input } => write!(
                f,
                "field collider input {input} has a singular or non-finite field transform"
            ),
        }
    }
}

/// One validated recipient-local observation entering the recombination law.
///
/// `velocity_uv_per_second` has already been mapped by
/// `linear(inverse(R) * D)`; translation never reaches a vector.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ColliderInputSample {
    pub velocity_uv_per_second: [f32; 2],
    pub confidence: f32,
    pub visibility: f32,
}

impl ColliderInputSample {
    /// Validate one mapped observation. Any non-finite or out-of-range
    /// component removes the whole sample; a partially trusted reading would
    /// let one hostile component steer a mode that mixes both.
    pub fn validated(self) -> Option<Self> {
        let [x, y] = self.velocity_uv_per_second;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        if x.abs() > MOTION_MAX_UV_PER_SECOND || y.abs() > MOTION_MAX_UV_PER_SECOND {
            return None;
        }
        if !self.confidence.is_finite() || !self.visibility.is_finite() {
            return None;
        }
        if !(0.0..=1.0).contains(&self.confidence) || !(0.0..=1.0).contains(&self.visibility) {
            return None;
        }
        Some(self)
    }
}

/// Squared-magnitude floor below which a direction is not a direction.
const COLLIDER_EPSILON: f32 = 1e-12;

/// The complete v1 recombination law over two validated recipient-local
/// vectors. This is the independent CPU reference `motion_collide.wgsl` is
/// measured against, expression for expression.
pub fn collide_vectors(mode: FieldColliderMode, a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let d = [a[0] - b[0], a[1] - b[1]];
    let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    let raw = match mode {
        FieldColliderMode::Sum => [a[0] + b[0], a[1] + b[1]],
        FieldColliderMode::Difference => d,
        FieldColliderMode::Curl => [-d[1], d[0]],
        FieldColliderMode::Projection => {
            let bb = b[0].mul_add(b[0], b[1] * b[1]);
            if bb <= COLLIDER_EPSILON {
                [0.0, 0.0]
            } else {
                let ab = a[0].mul_add(b[0], a[1] * b[1]);
                let scale = ab / bb;
                [b[0] * scale, b[1] * scale]
            }
        }
        FieldColliderMode::CollisionBoundary => {
            let dd = d[0].mul_add(d[0], d[1] * d[1]);
            if dd <= COLLIDER_EPSILON {
                m
            } else {
                // Remove the mean flow normal to disagreement.
                let md = m[0].mul_add(d[0], m[1] * d[1]);
                let scale = md / dd;
                [m[0] - d[0] * scale, m[1] - d[1] * scale]
            }
        }
    };
    [clamp_motion_velocity(raw[0]), clamp_motion_velocity(raw[1])]
}

/// The complete derived sample for one cell.
///
/// Both inputs must be present and validated. A missing or invalid input yields
/// the exact invalid/zero sample — it never reuses the surviving input and
/// never reuses a prior derived field, because either would present an
/// observation the collider did not make.
///
/// Confidence and visibility are componentwise minima. The Faraday gate then
/// applies threshold/softness/occlusion exactly once, downstream, in
/// `motion_apply.wgsl` and `motion_refresh.wgsl` — this function never
/// pre-applies it.
pub fn collide_motion_samples(
    mode: FieldColliderMode,
    a: Option<ColliderInputSample>,
    b: Option<ColliderInputSample>,
) -> MotionVectorSample {
    let (Some(a), Some(b)) = (
        a.and_then(ColliderInputSample::validated),
        b.and_then(ColliderInputSample::validated),
    ) else {
        return MotionVectorSample::default();
    };
    MotionVectorSample {
        velocity_uv_per_second: collide_vectors(
            mode,
            a.velocity_uv_per_second,
            b.velocity_uv_per_second,
        ),
        confidence: a.confidence.min(b.confidence),
        visibility: a.visibility.min(b.visibility),
    }
}

/// Peak procedural field speed in UV per second.
///
/// Eight is one eighth of [`MOTION_MAX_UV_PER_SECOND`]: strong enough that a
/// full-amount Faraday advection visibly moves the carrier, small enough that
/// the multi-octave Curl sum and the frequency-scaled Contour tangent stay
/// inside the canonical velocity range at ordinary settings instead of living
/// on the clamp.
pub const PROCEDURAL_FIELD_MAX_SPEED: f32 = 8.0;

const TAU: f32 = std::f32::consts::TAU;

/// The three frozen Curl octaves: wave vector, phase speed, phase offset.
/// These constants are part of the field law — the WGSL twin carries the same
/// numbers character for character.
const CURL_OCTAVES: [([f32; 2], f32, f32); 3] = [
    ([1.0, 0.618], 1.0, 0.0),
    ([-1.618, 1.0], -0.5, 0.25),
    ([0.786, -1.376], 0.25, 0.5),
];

/// Map the authored unit scale onto 1–16 spatial cycles across the frame.
pub fn procedural_field_frequency(scale: f32) -> f32 {
    1.0 + unit(scale, 0.5) * 15.0
}

/// The complete B2 procedural field law for one cell.
///
/// This is the independent CPU reference `motion_procedural.wgsl` is measured
/// against, expression for expression. `uv` addresses the cell centre in
/// `[0, 1]` field space, `cell_step_uv` is one field cell expressed in UV (the
/// Contour gradient's tap distance), and `time_seconds` is *program* time from
/// the shared frame-plan context — never wall time, so live and offline agree
/// by construction and Pause holds the field still.
///
/// `image` observes the recipient's picture as covered premultiplied linear
/// RGBA — exactly the quantity `covered_source_linear` computes — so hostile
/// RGB hidden behind zero coverage steers nothing. The closure is consulted
/// only by Contour and Chroma; the four pure kinds never call it.
///
/// Curl is the analytic curl of a three-octave sinusoidal stream function, so
/// it is divergence-free by construction rather than by numerical accident.
/// The pure kinds report fully open gates; Contour and Chroma report an honest
/// gradient/saturation confidence so flat content contributes nothing.
pub fn procedural_field_sample(
    kind: ProceduralFieldKind,
    uv: [f32; 2],
    cell_step_uv: [f32; 2],
    time_seconds: f32,
    params: ProceduralFieldParams,
    image: &dyn Fn([f32; 2]) -> [f32; 4],
) -> MotionVectorSample {
    let params = params.sanitized();
    let freq = procedural_field_frequency(params.scale);
    let phase = finite_or(time_seconds, 0.0).max(0.0) * params.rate;
    let amp = PROCEDURAL_FIELD_MAX_SPEED;
    let p = [uv[0] - 0.5, uv[1] - 0.5];
    let luma_at = |uv: [f32; 2]| {
        let c = image(uv);
        c[0].mul_add(0.2126, c[1].mul_add(0.7152, c[2] * 0.0722))
    };
    let (velocity, confidence) = match kind {
        ProceduralFieldKind::Curl => {
            // v = (dpsi/dy, -dpsi/dx) of
            // psi = sum(0.5^o / (TAU * freq) * sin(TAU * (freq * k.uv + c + s * phase))).
            let mut gradient = [0.0_f32, 0.0_f32];
            let mut weight = 1.0_f32;
            for (k, speed, offset) in CURL_OCTAVES {
                let argument =
                    TAU * (freq * k[0].mul_add(uv[0], k[1] * uv[1]) + offset + speed * phase);
                let ring = weight * argument.cos();
                gradient[0] += ring * k[0];
                gradient[1] += ring * k[1];
                weight *= 0.5;
            }
            ([amp * gradient[1], -amp * gradient[0]], 1.0)
        }
        ProceduralFieldKind::Radial | ProceduralFieldKind::Spiral => {
            let r = p[0].hypot(p[1]);
            if r <= 1.0e-6 {
                ([0.0, 0.0], 1.0)
            } else {
                let outward = [p[0] / r, p[1] / r];
                let direction = if kind == ProceduralFieldKind::Radial {
                    outward
                } else {
                    // The 45-degree pitch between outward and tangential.
                    [
                        (outward[0] - outward[1]) * std::f32::consts::FRAC_1_SQRT_2,
                        (outward[1] + outward[0]) * std::f32::consts::FRAC_1_SQRT_2,
                    ]
                };
                let ring = (TAU * freq.mul_add(r, -phase)).cos();
                ([amp * direction[0] * ring, amp * direction[1] * ring], 1.0)
            }
        }
        ProceduralFieldKind::Weave => (
            [
                amp * (TAU * freq.mul_add(uv[1], phase)).sin(),
                amp * 0.25 * (TAU * freq.mul_add(uv[0], -phase)).sin(),
            ],
            1.0,
        ),
        ProceduralFieldKind::Contour => {
            let gx = (luma_at([uv[0] + cell_step_uv[0], uv[1]])
                - luma_at([uv[0] - cell_step_uv[0], uv[1]]))
                * 0.5;
            let gy = (luma_at([uv[0], uv[1] + cell_step_uv[1]])
                - luma_at([uv[0], uv[1] - cell_step_uv[1]]))
                * 0.5;
            // Perpendicular to the luma gradient: flow along the isoline.
            let tangent = [-gy, gx];
            let swing = (TAU * phase).cos();
            (
                [
                    amp * tangent[0] * freq * swing,
                    amp * tangent[1] * freq * swing,
                ],
                unit(gx.hypot(gy) * 8.0, 0.0),
            )
        }
        ProceduralFieldKind::Chroma => {
            let c = image(uv);
            let i = 0.596f32.mul_add(c[0], (-0.274f32).mul_add(c[1], -0.322 * c[2]));
            let q = 0.211f32.mul_add(c[0], (-0.523f32).mul_add(c[1], 0.312 * c[2]));
            let angle = TAU * phase;
            let (sin, cos) = angle.sin_cos();
            let steered = [i.mul_add(cos, -(q * sin)), i.mul_add(sin, q * cos)];
            (
                [amp * 2.0 * steered[0], amp * 2.0 * steered[1]],
                unit(i.hypot(q) * 4.0, 0.0),
            )
        }
    };
    MotionVectorSample {
        velocity_uv_per_second: [
            clamp_motion_velocity(velocity[0]),
            clamp_motion_velocity(velocity[1]),
        ],
        confidence,
        visibility: 1.0,
    }
}

/// The flow-shaping event clock in ticks per second of program time.
///
/// A fixed law rather than an authored control: the plan's three shaping
/// amounts stay the whole authored surface, and `vector_trash` is a firing
/// probability per cell per tick.
pub const FLOW_TRASH_EVENT_HZ: f32 = 8.0;

/// Fixed domain constant separating the trash hash from every other stream.
const FLOW_TRASH_DOMAIN: u32 = 0x4d54_5253; // "MTRS"

/// The shared integer avalanche used by the deterministic per-cell laws.
/// Byte-identical to `effects.wgsl`'s `cellular_avalanche` so every backend
/// lands on the same path.
fn motion_avalanche(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// One deterministic unit sample for a trash cell, epoch, and lane. The
/// 24-bit mantissa path keeps CPU and every GPU backend bit-identical.
pub fn flow_trash_hash(cell: [i32; 2], epoch: u32, lane: u32) -> f32 {
    let seed = motion_avalanche(
        (cell[0] as u32)
            ^ (cell[1] as u32).wrapping_mul(0x9e37_79b9)
            ^ epoch.wrapping_mul(0x85eb_ca6b)
            ^ lane.wrapping_mul(0x27d4_eb2f)
            ^ FLOW_TRASH_DOMAIN,
    );
    (seed & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// The complete B2 flow-shaping law for one applied velocity.
///
/// This is the independent CPU reference the shaping block in
/// `motion_apply.wgsl` is measured against, expression for expression. It
/// operates on the *gated sampled* velocity — after confidence/visibility
/// gating, before the trajectory offset — so shaping modifies an applied
/// field and never manufactures motion where no admitted valid field exists.
///
/// `luma_gradient_per_texel` is the caller's central-difference observation of
/// the current image's covered luma, one texel out per axis; `output_px` is
/// the fragment position in output pixels; `time_seconds` is frame-plan
/// program time. Order is stretch, then repel, then trash, then the canonical
/// clamp, and the order is part of the law.
pub fn shape_flow_velocity(
    velocity: [f32; 2],
    uv: [f32; 2],
    luma_gradient_per_texel: [f32; 2],
    output_px: [f32; 2],
    time_seconds: f32,
    params: FlowShapingParams,
) -> [f32; 2] {
    let params = params.sanitized();
    let mut v = [finite_or(velocity[0], 0.0), finite_or(velocity[1], 0.0)];
    if params.stretch > 0.0 {
        let p = [uv[0] - 0.5, uv[1] - 0.5];
        let r = p[0].hypot(p[1]);
        if r > 1.0e-6 {
            let magnitude = v[0].hypot(v[1]);
            v[0] += p[0] / r * magnitude * params.stretch;
            v[1] += p[1] / r * magnitude * params.stretch;
        }
    }
    if params.edge_repel > 0.0 {
        let g = [
            finite_or(luma_gradient_per_texel[0], 0.0),
            finite_or(luma_gradient_per_texel[1], 0.0),
        ];
        let length = g[0].hypot(g[1]);
        if length > 1.0e-6 {
            // The push saturates at one full luma step per texel, so a razor
            // edge cannot launch an unbounded velocity before the clamp.
            let push = (length * 8.0).min(1.0) * PROCEDURAL_FIELD_MAX_SPEED * params.edge_repel;
            v[0] -= g[0] / length * push;
            v[1] -= g[1] / length * push;
        }
    }
    if params.vector_trash > 0.0 {
        let block = params.trash_block_size;
        let cell = [
            (finite_or(output_px[0], 0.0) / block).floor() as i32,
            (finite_or(output_px[1], 0.0) / block).floor() as i32,
        ];
        let epoch = (finite_or(time_seconds, 0.0).max(0.0) * FLOW_TRASH_EVENT_HZ) as u32;
        if flow_trash_hash(cell, epoch, 0) < params.vector_trash {
            let garbage = [
                flow_trash_hash(cell, epoch, 1).mul_add(2.0, -1.0),
                flow_trash_hash(cell, epoch, 2).mul_add(2.0, -1.0),
            ];
            let shove = 2.0 * PROCEDURAL_FIELD_MAX_SPEED;
            v[0] += garbage[0] * shove;
            v[1] += garbage[1] * shove;
        }
    }
    [clamp_motion_velocity(v[0]), clamp_motion_velocity(v[1])]
}

/// Byte-exact collider-specific resource delta for one admitted collider.
///
/// The two primitive input fields and the sole persistent carrier are already
/// charged by [`MotionResourcePlan::preflight`]; this plan carries only the
/// three surfaces the collider itself adds, so nothing is double counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FieldColliderResourcePlan {
    pub active_colliders: u32,
    pub grid: Option<MotionGrid>,
    pub derived_vector_bytes: u64,
    pub derived_gate_bytes: u64,
    pub transient_pair_bytes: u64,
    pub total_bytes: u64,
    /// Two low-resolution passes per admitted collider.
    pub low_resolution_passes: u32,
    /// Five nearest lookups per admitted collider: two in pass 1, three in
    /// pass 2.
    pub nearest_lookups: u32,
    /// The ordinary three-sampled-texture ceiling is unchanged by this block.
    pub max_sampled_textures_in_pass: u32,
}

impl FieldColliderResourcePlan {
    /// Preflight one composition's collider requests.
    ///
    /// `grids` carries the output grid of every admitted collider. An empty
    /// slice is the exact-M4 plan: all zeros, no pass, no surface.
    pub fn preflight(
        grids: &[MotionGrid],
        limits: MotionDeviceLimits,
    ) -> Result<Self, MotionPlanError> {
        let mut plan = Self::default();
        for grid in grids {
            plan.active_colliders = plan
                .active_colliders
                .checked_add(1)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            if plan.active_colliders > MOTION_MAX_ACTIVE_COLLIDERS {
                return Err(MotionPlanError::TooManyColliders {
                    count: plan.active_colliders,
                    limit: MOTION_MAX_ACTIVE_COLLIDERS,
                });
            }
            if grid.width == 0
                || grid.height == 0
                || grid.width > MOTION_FIELD_MAX_EDGE
                || grid.height > MOTION_FIELD_MAX_EDGE
            {
                return Err(MotionPlanError::FieldEdge {
                    dimensions: [grid.width, grid.height],
                    limit: MOTION_FIELD_MAX_EDGE,
                });
            }
            if grid.width > limits.max_texture_dimension_2d
                || grid.height > limits.max_texture_dimension_2d
            {
                return Err(MotionPlanError::DeviceTextureDimension {
                    dimensions: [grid.width, grid.height],
                    limit: limits.max_texture_dimension_2d,
                });
            }
            let cells = u64::from(grid.width)
                .checked_mul(u64::from(grid.height))
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            if cells != grid.vector_count || cells > MOTION_FIELD_MAX_VECTORS {
                return Err(MotionPlanError::VectorCount {
                    count: cells,
                    limit: MOTION_FIELD_MAX_VECTORS,
                });
            }
            let derived_vector_bytes = checked_bytes(cells, DERIVED_VECTOR_BYTES_PER_CELL)?;
            let derived_gate_bytes = checked_bytes(cells, DERIVED_GATE_BYTES_PER_CELL)?;
            let transient_pair_bytes = checked_bytes(cells, TRANSIENT_PAIR_BYTES_PER_CELL)?;
            let field_bytes = derived_vector_bytes
                .checked_add(derived_gate_bytes)
                .and_then(|value| value.checked_add(transient_pair_bytes))
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            // One collider's complete working set is one field's worth of
            // resource and is bounded by the same per-field ceiling.
            if field_bytes > MOTION_FIELD_MAX_BYTES {
                return Err(MotionPlanError::FieldBytes {
                    bytes: field_bytes,
                    limit: MOTION_FIELD_MAX_BYTES,
                });
            }
            plan.grid = Some(*grid);
            plan.derived_vector_bytes = plan
                .derived_vector_bytes
                .checked_add(derived_vector_bytes)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            plan.derived_gate_bytes = plan
                .derived_gate_bytes
                .checked_add(derived_gate_bytes)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            plan.transient_pair_bytes = plan
                .transient_pair_bytes
                .checked_add(transient_pair_bytes)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            plan.low_resolution_passes = plan
                .low_resolution_passes
                .checked_add(2)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            plan.nearest_lookups = plan
                .nearest_lookups
                .checked_add(5)
                .ok_or(MotionPlanError::ArithmeticOverflow)?;
            plan.max_sampled_textures_in_pass = plan.max_sampled_textures_in_pass.max(3);
        }
        plan.total_bytes = plan
            .derived_vector_bytes
            .checked_add(plan.derived_gate_bytes)
            .and_then(|value| value.checked_add(plan.transient_pair_bytes))
            .ok_or(MotionPlanError::ArithmeticOverflow)?;
        if plan.total_bytes > limits.max_motion_bytes {
            return Err(MotionPlanError::AggregateBytes {
                bytes: plan.total_bytes,
                limit: limits.max_motion_bytes,
            });
        }
        Ok(plan)
    }

    /// The exact per-cell charge this plan represents. Zero with no collider.
    pub const fn bytes_per_cell(self) -> u64 {
        if self.active_colliders == 0 {
            0
        } else {
            FIELD_COLLIDER_BYTES_PER_CELL
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generous_limits() -> MotionDeviceLimits {
        MotionDeviceLimits::new(8_192, 256 * 1024 * 1024)
    }

    // -----------------------------------------------------------------
    // Field Collider v1
    // -----------------------------------------------------------------

    fn collider_donor(id: u64, position: u32) -> MotionDonor {
        MotionDonor::Selected {
            layer_id: StableLayerId::new(id).unwrap(),
            saved_position: SavedLayerPosition::new(position).unwrap(),
        }
    }

    fn armed_collider(mode: FieldColliderMode) -> FieldColliderParams {
        FieldColliderParams {
            enabled: true,
            mode,
            input_a: collider_donor(11, 0),
            input_b: collider_donor(22, 1),
            ..FieldColliderParams::default()
        }
    }

    fn collider_sample(
        velocity: [f32; 2],
        confidence: f32,
        visibility: f32,
    ) -> ColliderInputSample {
        ColliderInputSample {
            velocity_uv_per_second: velocity,
            confidence,
            visibility,
        }
    }

    #[test]
    fn field_collider_vocabularies_are_closed_and_their_codes_are_append_only() {
        assert_eq!(FIELD_COLLIDER_ALGORITHM_VERSION, 1);
        let codes: Vec<u32> = FieldColliderMode::ALL
            .iter()
            .map(|mode| mode.code())
            .collect();
        assert_eq!(codes, vec![0, 1, 2, 3, 4]);
        assert_eq!(FieldColliderMode::default(), FieldColliderMode::Sum);

        // The motion boundary deliberately does NOT mint a fourth table: its
        // codes are the frozen image-boundary numbering, so `1` means Mirror
        // everywhere in the program.
        let boundary_codes: Vec<u32> = MotionBoundaryMode::ALL
            .iter()
            .map(|boundary| boundary.code())
            .collect();
        assert_eq!(boundary_codes, vec![0, 1, 2, 3]);
        assert_eq!(
            MotionBoundaryMode::default(),
            MotionBoundaryMode::Transparent
        );
        assert_eq!(
            MotionBoundaryMode::Transparent.code(),
            crate::visual_rack::DisplaceBoundary::Transparent.code()
        );
        assert_eq!(
            MotionBoundaryMode::Mirror.code(),
            crate::visual_rack::DisplaceBoundary::Mirror.code()
        );
        assert_eq!(
            MotionBoundaryMode::Wrap.code(),
            crate::visual_rack::DisplaceBoundary::Wrap.code()
        );
        assert_eq!(
            MotionBoundaryMode::Hold.code(),
            crate::visual_rack::DisplaceBoundary::Hold.code()
        );
        assert_eq!(
            MotionBoundaryMode::Hold.code(),
            crate::symmetry::SymmetryBoundary::Hold.code()
        );

        assert_eq!(FieldColliderInput::A.index(), 0);
        assert_eq!(FieldColliderInput::B.index(), 1);
        assert_eq!(
            FieldColliderInput::from_index(0),
            Some(FieldColliderInput::A)
        );
        assert_eq!(
            FieldColliderInput::from_index(1),
            Some(FieldColliderInput::B)
        );
        assert_eq!(FieldColliderInput::from_index(2), None);
    }

    #[test]
    fn field_collider_modes_match_their_analytic_definitions() {
        let a = [3.0, 1.0];
        let b = [1.0, -1.0];
        // d = a - b = (2, 2); m = (a + b) / 2 = (2, 0).
        assert_eq!(collide_vectors(FieldColliderMode::Sum, a, b), [4.0, 0.0]);
        assert_eq!(
            collide_vectors(FieldColliderMode::Difference, a, b),
            [2.0, 2.0]
        );
        // Curl is d rotated a quarter turn: (-d.y, d.x).
        assert_eq!(collide_vectors(FieldColliderMode::Curl, a, b), [-2.0, 2.0]);
        // Projection: b * dot(a,b)/dot(b,b) = (1,-1) * 2/2 = (1,-1).
        assert_eq!(
            collide_vectors(FieldColliderMode::Projection, a, b),
            [1.0, -1.0]
        );
        // Collision boundary: m - d * dot(m,d)/dot(d,d) = (2,0) - (2,2)*4/8.
        assert_eq!(
            collide_vectors(FieldColliderMode::CollisionBoundary, a, b),
            [1.0, -1.0]
        );

        // Every mode is exactly zero on two zero inputs, and the two degenerate
        // guards return their documented value rather than a division by zero.
        for mode in FieldColliderMode::ALL {
            assert_eq!(collide_vectors(mode, [0.0, 0.0], [0.0, 0.0]), [0.0, 0.0]);
        }
        assert_eq!(
            collide_vectors(FieldColliderMode::Projection, [5.0, 7.0], [0.0, 0.0]),
            [0.0, 0.0],
            "a zero b has no direction to project onto"
        );
        assert_eq!(
            collide_vectors(FieldColliderMode::CollisionBoundary, [4.0, 6.0], [4.0, 6.0]),
            [4.0, 6.0],
            "two agreeing inputs have no disagreement normal to remove, so the mean survives"
        );
    }

    #[test]
    fn every_mode_clamps_into_the_canonical_velocity_range() {
        let extreme = MOTION_MAX_UV_PER_SECOND;
        // Sum is the mode that can overflow from two in-range inputs.
        assert_eq!(
            collide_vectors(
                FieldColliderMode::Sum,
                [extreme, extreme],
                [extreme, extreme]
            ),
            [extreme, extreme]
        );
        assert_eq!(
            collide_vectors(
                FieldColliderMode::Difference,
                [-extreme, -extreme],
                [extreme, extreme]
            ),
            [-extreme, -extreme]
        );
        assert_eq!(
            collide_vectors(
                FieldColliderMode::Curl,
                [extreme, -extreme],
                [-extreme, extreme]
            ),
            [extreme, extreme]
        );
        // The clamped value survives the canonical pack/unpack round trip,
        // which is the whole point of clamping to exactly that interval.
        for value in [extreme, -extreme, 0.0, 1.5] {
            let clamped = clamp_motion_velocity(value);
            let round_tripped = unpack_velocity(pack_velocity(clamped));
            assert!(
                (round_tripped - clamped).abs() <= extreme / f32::from(i16::MAX) + 1e-6,
                "{clamped} did not survive the canonical lattice"
            );
        }
        // A non-finite input lands on the DEFAULT, never on a clamped extreme.
        // That is the established `finite_or` idiom throughout this module: an
        // infinity is a broken observation, not a very fast one, and clamping
        // it to the maximum velocity would invent the strongest possible motion
        // out of a fault.
        assert_eq!(clamp_motion_velocity(f32::NAN), 0.0);
        assert_eq!(clamp_motion_velocity(f32::INFINITY), 0.0);
        assert_eq!(clamp_motion_velocity(f32::NEG_INFINITY), 0.0);
        // A finite over-range value IS clamped, because it is a real reading
        // that merely exceeds what the field contract can carry.
        assert_eq!(clamp_motion_velocity(extreme * 3.0), extreme);
        assert_eq!(clamp_motion_velocity(-extreme * 3.0), -extreme);
    }

    #[test]
    fn confidence_and_visibility_are_componentwise_minima_and_are_never_pre_gated() {
        let derived = collide_motion_samples(
            FieldColliderMode::Sum,
            Some(collider_sample([1.0, 2.0], 0.25, 0.9)),
            Some(collider_sample([3.0, 4.0], 0.75, 0.1)),
        );
        assert_eq!(derived.velocity_uv_per_second, [4.0, 6.0]);
        assert_eq!(derived.confidence, 0.25);
        assert_eq!(derived.visibility, 0.1);
        // The velocity is emphatically NOT scaled by either gate here: the
        // Faraday gate applies threshold/softness/occlusion exactly once,
        // downstream.
        assert_eq!(derived.velocity_uv_per_second, [4.0, 6.0]);
    }

    #[test]
    fn an_invalid_input_yields_the_exact_zero_sample_and_never_reuses_its_partner() {
        let good = collider_sample([7.0, -7.0], 1.0, 1.0);
        let zero = MotionVectorSample::default();
        for mode in FieldColliderMode::ALL {
            assert_eq!(collide_motion_samples(mode, Some(good), None), zero);
            assert_eq!(collide_motion_samples(mode, None, Some(good)), zero);
            assert_eq!(collide_motion_samples(mode, None, None), zero);
            for hostile in [
                collider_sample([f32::NAN, 0.0], 1.0, 1.0),
                collider_sample([0.0, f32::INFINITY], 1.0, 1.0),
                collider_sample([MOTION_MAX_UV_PER_SECOND * 2.0, 0.0], 1.0, 1.0),
                collider_sample([0.0, 0.0], f32::NAN, 1.0),
                collider_sample([0.0, 0.0], 1.0, 2.0),
                collider_sample([0.0, 0.0], -0.5, 1.0),
            ] {
                assert_eq!(
                    collide_motion_samples(mode, Some(good), Some(hostile)),
                    zero,
                    "{mode:?} reused a surviving input"
                );
                assert_eq!(
                    collide_motion_samples(mode, Some(hostile), Some(good)),
                    zero,
                    "{mode:?} reused a surviving input"
                );
            }
        }
        assert_eq!(
            ColliderInputSample::default().validated(),
            Some(ColliderInputSample::default()),
            "an all-zero observation is valid; it is simply stationary"
        );
    }

    #[test]
    fn every_boundary_law_resolves_or_removes_its_lookup() {
        // Transparent is inclusive on both edges and is the only removing law.
        assert_eq!(
            MotionBoundaryMode::Transparent.resolve([0.0, 1.0]),
            Some([0.0, 1.0])
        );
        assert_eq!(MotionBoundaryMode::Transparent.resolve([-0.001, 0.5]), None);
        assert_eq!(MotionBoundaryMode::Transparent.resolve([0.5, 1.001]), None);

        assert_eq!(
            MotionBoundaryMode::Hold.resolve([-3.0, 4.0]),
            Some([0.0, 1.0])
        );
        assert_eq!(
            MotionBoundaryMode::Wrap.resolve([1.25, -0.25]),
            Some([0.25, 0.75])
        );
        // Mirror is the period-two triangular map: 1.25 folds to 0.75.
        assert_eq!(
            MotionBoundaryMode::Mirror.resolve([1.25, -0.25]),
            Some([0.75, 0.25])
        );
        assert_eq!(
            MotionBoundaryMode::Mirror.resolve([2.0, 3.0]),
            Some([0.0, 1.0])
        );

        // A non-finite coordinate is removed by EVERY law, including the three
        // that otherwise always produce a sample.
        for boundary in MotionBoundaryMode::ALL {
            assert_eq!(boundary.resolve([f32::NAN, 0.5]), None, "{boundary:?}");
            assert_eq!(boundary.resolve([0.5, f32::INFINITY]), None, "{boundary:?}");
            // Whatever survives is inside the closed unit interval.
            if let Some([x, y]) = boundary.resolve([7.3, -4.1]) {
                assert!((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y));
            }
        }
    }

    #[test]
    fn collider_admission_refuses_alias_missing_unselected_master_and_zero_transplant() {
        let admitted = armed_collider(FieldColliderMode::Curl);
        assert_eq!(
            admitted.admission(),
            FieldColliderAdmission::Admitted {
                input_a: StableLayerId::new(11).unwrap(),
                input_b: StableLayerId::new(22).unwrap(),
            }
        );
        assert!(admitted.is_admitted());
        assert_eq!(
            admitted.admission().diagnostic(),
            FieldColliderDiagnostic::None
        );

        // Disabled is exact M4 and carries no fault.
        let disabled = FieldColliderParams::default();
        assert!(disabled.is_exact_m4());
        assert!(!disabled.is_admitted());
        assert_eq!(
            disabled.admission().diagnostic(),
            FieldColliderDiagnostic::Disabled
        );
        assert!(!FieldColliderDiagnostic::Disabled.is_fault());
        assert!(!FieldColliderDiagnostic::None.is_fault());

        // A and B may never alias each other.
        let mut aliased = admitted;
        aliased.input_b = aliased.input_a;
        assert_eq!(
            aliased.admission().diagnostic(),
            FieldColliderDiagnostic::AliasedInputs
        );
        assert!(FieldColliderDiagnostic::AliasedInputs.is_fault());

        // Aliasing by layer id alone, even at a different saved position.
        let mut aliased_position = admitted;
        aliased_position.input_b = collider_donor(11, 5);
        assert_eq!(
            aliased_position.admission().diagnostic(),
            FieldColliderDiagnostic::AliasedInputs
        );

        // A tombstone never rebinds; it refuses by name and by slot.
        let mut missing_b = admitted;
        missing_b.input_b = MotionDonor::Missing {
            saved_position: SavedLayerPosition::new(1).unwrap(),
        };
        assert_eq!(
            missing_b.admission().diagnostic(),
            FieldColliderDiagnostic::InputMissing {
                input: FieldColliderInput::B
            }
        );
        let mut unselected_a = admitted;
        unselected_a.input_a = MotionDonor::None;
        assert_eq!(
            unselected_a.admission().diagnostic(),
            FieldColliderDiagnostic::InputUnselected {
                input: FieldColliderInput::A
            }
        );

        // Scope-level refusals, answered by the one shared predicate.
        let mut params = MotionParams {
            collider: admitted,
            ..MotionParams::default()
        };
        params.transplant.amount = 0.5;
        assert!(matches!(
            params.collider_admission(false),
            FieldColliderAdmission::Admitted { .. }
        ));
        assert_eq!(
            params.collider_admission(true).diagnostic(),
            FieldColliderDiagnostic::MasterRecipient
        );
        params.transplant.amount = 0.0;
        assert_eq!(
            params.collider_admission(false).diagnostic(),
            FieldColliderDiagnostic::NoActiveTransplant
        );
        // Authored inertness outranks every environmental fault, so a disabled
        // block never accuses its scope of a problem it does not have.
        params.collider.enabled = false;
        assert_eq!(
            params.collider_admission(true).diagnostic(),
            FieldColliderDiagnostic::Disabled
        );
    }

    #[test]
    fn a_collider_input_may_name_its_own_recipient() {
        // Only A-aliases-B is refused. A layer colliding its own field against
        // another layer's is authored, not a cycle.
        let recipient = StableLayerId::new(11).unwrap();
        let params = FieldColliderParams {
            enabled: true,
            input_a: collider_donor(recipient.get(), 0),
            input_b: collider_donor(22, 1),
            ..FieldColliderParams::default()
        };
        assert_eq!(
            params.admission(),
            FieldColliderAdmission::Admitted {
                input_a: recipient,
                input_b: StableLayerId::new(22).unwrap(),
            }
        );
    }

    #[test]
    fn the_collider_block_sanitizes_and_survives_motion_sanitization() {
        let hostile = FieldColliderParams {
            algorithm_version: 99,
            ..armed_collider(FieldColliderMode::Projection)
        };
        assert_eq!(
            hostile.sanitized().algorithm_version,
            FIELD_COLLIDER_ALGORITHM_VERSION
        );
        // Sanitizing preserves every authored field except the pinned version.
        assert_eq!(hostile.sanitized().mode, FieldColliderMode::Projection);
        assert_eq!(hostile.sanitized().input_a, hostile.input_a);
        assert_eq!(hostile.sanitized().input_b, hostile.input_b);

        let params = MotionParams {
            collider: hostile,
            ..MotionParams::default()
        };
        assert_eq!(
            params.sanitized().collider.algorithm_version,
            FIELD_COLLIDER_ALGORITHM_VERSION
        );
        assert_eq!(params.sanitized().collider.input_b, hostile.input_b);

        // The default block is exactly M4 and does not disturb is_exact_zero.
        assert!(MotionParams::default().is_exact_zero());
        assert!(MotionParams::default().collider.is_exact_m4());
        assert_eq!(
            FieldColliderParams::default().input(FieldColliderInput::A),
            MotionDonor::None
        );

        // Slot identity is a named field: writing A never disturbs B.
        let mut slots = FieldColliderParams::default();
        *slots.input_mut(FieldColliderInput::B) = collider_donor(7, 3);
        assert_eq!(slots.input(FieldColliderInput::A), MotionDonor::None);
        assert_eq!(slots.input(FieldColliderInput::B), collider_donor(7, 3));
        *slots.input_mut(FieldColliderInput::A) = collider_donor(8, 4);
        assert_eq!(slots.input(FieldColliderInput::B), collider_donor(7, 3));
    }

    #[test]
    fn the_collider_ledger_is_exactly_twenty_bytes_per_cell_and_admits_one_collider() {
        let limits = generous_limits();
        let grid = MotionGrid::for_source([640, 480], MotionLatticeQuality::Live).unwrap();
        let cells = grid.vector_count;

        let empty = FieldColliderResourcePlan::preflight(&[], limits).unwrap();
        assert_eq!(empty, FieldColliderResourcePlan::default());
        assert_eq!(empty.total_bytes, 0);
        assert_eq!(empty.bytes_per_cell(), 0);
        assert_eq!(empty.low_resolution_passes, 0);

        let plan = FieldColliderResourcePlan::preflight(&[grid], limits).unwrap();
        assert_eq!(plan.active_colliders, 1);
        assert_eq!(plan.grid, Some(grid));
        // 8 derived-vector + 4 derived-gate + 8 transient-pair = 20.
        assert_eq!(plan.derived_vector_bytes, cells * 8);
        assert_eq!(plan.derived_gate_bytes, cells * 4);
        assert_eq!(plan.transient_pair_bytes, cells * 8);
        assert_eq!(plan.total_bytes, cells * 20);
        assert_eq!(plan.bytes_per_cell(), FIELD_COLLIDER_BYTES_PER_CELL);
        assert_eq!(FIELD_COLLIDER_BYTES_PER_CELL, 20);
        // Two low-resolution passes and five nearest lookups per collider, with
        // the ordinary three-sampled-texture ceiling unchanged.
        assert_eq!(plan.low_resolution_passes, 2);
        assert_eq!(plan.nearest_lookups, 5);
        assert_eq!(plan.max_sampled_textures_in_pass, 3);

        // The one-collider cap.
        assert_eq!(
            FieldColliderResourcePlan::preflight(&[grid, grid], limits),
            Err(MotionPlanError::TooManyColliders {
                count: 2,
                limit: MOTION_MAX_ACTIVE_COLLIDERS,
            })
        );
        assert_eq!(MOTION_MAX_ACTIVE_COLLIDERS, 1);
    }

    #[test]
    fn the_collider_ledger_rejects_one_byte_over_every_independent_bound() {
        let limits = generous_limits();
        // Exactly at the per-field byte ceiling, then one cell over it.
        let admitted_cells = MOTION_FIELD_MAX_BYTES / FIELD_COLLIDER_BYTES_PER_CELL;
        let width = 1_024_u32;
        let height = u32::try_from(admitted_cells / u64::from(width)).unwrap();
        let at_cap = MotionGrid {
            width,
            height,
            block_pixels: 8,
            vector_count: u64::from(width) * u64::from(height),
        };
        let plan = FieldColliderResourcePlan::preflight(&[at_cap], limits).unwrap();
        assert!(plan.total_bytes <= MOTION_FIELD_MAX_BYTES);

        let one_row_over = MotionGrid {
            height: height + 1,
            vector_count: u64::from(width) * u64::from(height + 1),
            ..at_cap
        };
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[one_row_over], limits),
            Err(MotionPlanError::FieldBytes { .. })
        ));

        // One over the absolute edge bound.
        let over_edge = MotionGrid {
            width: MOTION_FIELD_MAX_EDGE + 1,
            height: 1,
            block_pixels: 8,
            vector_count: u64::from(MOTION_FIELD_MAX_EDGE) + 1,
        };
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[over_edge], limits),
            Err(MotionPlanError::FieldEdge { .. })
        ));

        // A zero extent is refused rather than silently producing no surface.
        let empty_grid = MotionGrid {
            width: 0,
            height: 4,
            block_pixels: 8,
            vector_count: 0,
        };
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[empty_grid], limits),
            Err(MotionPlanError::FieldEdge { .. })
        ));

        // A grid whose declared count disagrees with its extent is refused,
        // never trusted.
        let inconsistent = MotionGrid {
            width: 8,
            height: 8,
            block_pixels: 8,
            vector_count: 63,
        };
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[inconsistent], limits),
            Err(MotionPlanError::VectorCount { .. })
        ));

        // The device texture-edge bound binds independently of the motion cap.
        let small_device = MotionDeviceLimits::new(64, 256 * 1024 * 1024);
        let grid = MotionGrid::for_source([1_920, 1_080], MotionLatticeQuality::Live).unwrap();
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[grid], small_device),
            Err(MotionPlanError::DeviceTextureDimension { .. })
        ));

        // And so does the aggregate motion byte budget.
        let tight = MotionDeviceLimits {
            max_motion_bytes: 1,
            ..generous_limits()
        };
        assert!(matches!(
            FieldColliderResourcePlan::preflight(&[grid], tight),
            Err(MotionPlanError::AggregateBytes { .. })
        ));
    }

    #[test]
    fn a_collider_diagnostic_names_its_slot_and_carries_no_host_detail() {
        for diagnostic in [
            FieldColliderDiagnostic::Disabled,
            FieldColliderDiagnostic::AliasedInputs,
            FieldColliderDiagnostic::MasterRecipient,
            FieldColliderDiagnostic::NoActiveTransplant,
            FieldColliderDiagnostic::InputMissing {
                input: FieldColliderInput::A,
            },
            FieldColliderDiagnostic::InputUnselected {
                input: FieldColliderInput::B,
            },
            FieldColliderDiagnostic::InputFieldUnavailable {
                input: FieldColliderInput::A,
            },
            FieldColliderDiagnostic::SingularTransform {
                input: FieldColliderInput::B,
            },
        ] {
            let rendered = diagnostic.to_string();
            assert!(!rendered.is_empty());
            // Authored identity only: no path separator, no drive letter, no
            // filesystem metadata.
            assert!(!rendered.contains('/'), "{rendered}");
            assert!(!rendered.contains('\\'), "{rendered}");
            assert!(!rendered.contains(':'), "{rendered}");
        }
        assert!(FieldColliderDiagnostic::InputMissing {
            input: FieldColliderInput::A
        }
        .to_string()
        .contains(" A "));
        assert!(FieldColliderDiagnostic::SingularTransform {
            input: FieldColliderInput::B
        }
        .to_string()
        .contains(" B "));
        assert_eq!(FieldColliderInput::A.to_string(), "A");
        assert_eq!(FieldColliderInput::B.to_string(), "B");
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
            required_as_study_input: false,
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
            required_as_study_input: false,
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
            required_as_study_input: false,
        };
        let donor = MotionScopeResourceRequest {
            source_dimensions: [1280, 720],
            output_dimensions: [1920, 1080],
            params: MotionParams::default(),
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: true,
            required_as_garden_signal: false,
            required_as_study_input: false,
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
            required_as_study_input: false,
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

    // -----------------------------------------------------------------
    // B2 procedural fields
    // -----------------------------------------------------------------

    /// Image closure for the four pure kinds: consulting it at all is the
    /// defect under test.
    fn forbidden_image() -> impl Fn([f32; 2]) -> [f32; 4] {
        |_uv: [f32; 2]| -> [f32; 4] { panic!("a pure procedural kind observed the image") }
    }

    fn sample_pure(
        kind: ProceduralFieldKind,
        uv: [f32; 2],
        time: f32,
        params: ProceduralFieldParams,
    ) -> MotionVectorSample {
        procedural_field_sample(kind, uv, [0.01, 0.01], time, params, &forbidden_image())
    }

    #[test]
    fn procedural_kind_codes_and_wire_keys_are_permanent() {
        let expected = [
            (ProceduralFieldKind::Curl, 0, "procedural_curl", false),
            (ProceduralFieldKind::Radial, 1, "procedural_radial", false),
            (ProceduralFieldKind::Spiral, 2, "procedural_spiral", false),
            (ProceduralFieldKind::Contour, 3, "procedural_contour", true),
            (ProceduralFieldKind::Chroma, 4, "procedural_chroma", true),
            (ProceduralFieldKind::Weave, 5, "procedural_weave", false),
        ];
        for (index, (kind, code, key, reads_image)) in expected.into_iter().enumerate() {
            assert_eq!(ProceduralFieldKind::ALL[index], kind);
            assert_eq!(kind.code(), code);
            assert_eq!(kind.source_key(), key);
            assert_eq!(kind.reads_image(), reads_image);
        }
        // The origin signature codes are append-only: the original four keep
        // 0-3 and the six kinds occupy 4-9 without collision, so a kind change
        // re-prepares bind groups instead of silently reusing them.
        assert_eq!(MotionFieldOrigin::None.signature_code(), 0);
        assert_eq!(MotionFieldOrigin::CodecVectors.signature_code(), 1);
        assert_eq!(MotionFieldOrigin::Lattice.signature_code(), 2);
        assert_eq!(MotionFieldOrigin::LatticeFallback.signature_code(), 3);
        let mut seen = std::collections::BTreeSet::new();
        for kind in ProceduralFieldKind::ALL {
            let code = MotionFieldOrigin::Procedural(kind).signature_code();
            assert!((4..=9).contains(&code));
            assert!(seen.insert(code), "duplicate signature code {code}");
        }
    }

    #[test]
    fn procedural_source_resolves_at_every_scope_without_codec_or_fallback() {
        for kind in ProceduralFieldKind::ALL {
            for is_master in [false, true] {
                for codec_available in [false, true] {
                    let decision = resolve_motion_source(
                        MotionFieldSource::Procedural(kind),
                        is_master,
                        codec_available,
                    );
                    assert_eq!(decision.origin, MotionFieldOrigin::Procedural(kind));
                    assert_eq!(decision.diagnostic, MotionSourceDiagnostic::None);
                }
            }
        }
    }

    #[test]
    fn procedural_params_sanitize_to_neutral_values_not_clamped_extremes() {
        let hostile = ProceduralFieldParams {
            scale: f32::NAN,
            rate: f32::INFINITY,
        }
        .sanitized();
        assert_eq!(hostile.scale, 0.5);
        assert_eq!(hostile.rate, 0.25);
        let clamped = ProceduralFieldParams {
            scale: 2.0,
            rate: -9.0,
        }
        .sanitized();
        assert_eq!(clamped.scale, 1.0);
        assert_eq!(clamped.rate, -2.0);
        // The params ride MotionParams::sanitized like every other block.
        let motion = MotionParams {
            procedural: ProceduralFieldParams {
                scale: f32::NAN,
                rate: 5.0,
            },
            ..MotionParams::default()
        }
        .sanitized();
        assert_eq!(motion.procedural.scale, 0.5);
        assert_eq!(motion.procedural.rate, 2.0);
    }

    #[test]
    fn pure_kinds_never_observe_the_image_and_open_their_gates_fully() {
        for kind in [
            ProceduralFieldKind::Curl,
            ProceduralFieldKind::Radial,
            ProceduralFieldKind::Spiral,
            ProceduralFieldKind::Weave,
        ] {
            let sample = sample_pure(kind, [0.3, 0.7], 1.25, ProceduralFieldParams::default());
            assert_eq!(sample.confidence, 1.0, "{kind} must open its gate");
            assert_eq!(sample.visibility, 1.0);
            assert!(sample.velocity_uv_per_second[0].is_finite());
            assert!(sample.velocity_uv_per_second[1].is_finite());
        }
    }

    #[test]
    fn radial_pulses_outward_on_the_ring_law_and_is_still_at_the_centre() {
        let params = ProceduralFieldParams {
            scale: 0.0, // freq = 1
            rate: 0.0,
        };
        // At the centre there is no direction, so there is no velocity.
        let centre = sample_pure(ProceduralFieldKind::Radial, [0.5, 0.5], 0.0, params);
        assert_eq!(centre.velocity_uv_per_second, [0.0, 0.0]);
        // On the +x axis at radius r the law is amp * cos(TAU * freq * r).
        let r = 0.25_f32;
        let sample = sample_pure(ProceduralFieldKind::Radial, [0.5 + r, 0.5], 0.0, params);
        let expected = PROCEDURAL_FIELD_MAX_SPEED * (TAU * r).cos();
        assert!((sample.velocity_uv_per_second[0] - expected).abs() < 1.0e-4);
        assert!(sample.velocity_uv_per_second[1].abs() < 1.0e-4);
    }

    #[test]
    fn spiral_is_the_radial_ring_pitched_forty_five_degrees() {
        let params = ProceduralFieldParams {
            scale: 0.0,
            rate: 0.0,
        };
        let r = 0.2_f32;
        let radial = sample_pure(ProceduralFieldKind::Radial, [0.5 + r, 0.5], 0.0, params);
        let spiral = sample_pure(ProceduralFieldKind::Spiral, [0.5 + r, 0.5], 0.0, params);
        let magnitude = |v: [f32; 2]| v[0].hypot(v[1]);
        assert!(
            (magnitude(radial.velocity_uv_per_second) - magnitude(spiral.velocity_uv_per_second))
                .abs()
                < 1.0e-3
        );
        // On the +x axis the outward direction is +x; the pitched direction
        // splits equally between +x and +y.
        assert!(
            (spiral.velocity_uv_per_second[0] - spiral.velocity_uv_per_second[1]).abs() < 1.0e-3
        );
    }

    #[test]
    fn weave_is_orthogonal_sinusoidal_shear() {
        let params = ProceduralFieldParams {
            scale: 0.0,
            rate: 0.0,
        };
        let uv = [0.15, 0.4];
        let sample = sample_pure(ProceduralFieldKind::Weave, uv, 0.0, params);
        let expected_x = PROCEDURAL_FIELD_MAX_SPEED * (TAU * uv[1]).sin();
        let expected_y = PROCEDURAL_FIELD_MAX_SPEED * 0.25 * (TAU * uv[0]).sin();
        assert!((sample.velocity_uv_per_second[0] - expected_x).abs() < 1.0e-4);
        assert!((sample.velocity_uv_per_second[1] - expected_y).abs() < 1.0e-4);
    }

    #[test]
    fn curl_is_numerically_divergence_free() {
        // Base frequency: the analytic claim holds at every scale, but a
        // central-difference estimate of an oscillatory f32 field carries
        // O(amp * omega^3 * eps^2) truncation error, so the numeric check is
        // meaningful only where that term stays far below the field peak.
        let params = ProceduralFieldParams {
            scale: 0.0,
            rate: 1.0,
        };
        let epsilon = 1.0e-3_f32;
        for (x, y) in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.1), (0.35, 0.9)] {
            let at = |uv: [f32; 2]| {
                sample_pure(ProceduralFieldKind::Curl, uv, 2.0, params).velocity_uv_per_second
            };
            let dvx_dx = (at([x + epsilon, y])[0] - at([x - epsilon, y])[0]) / (2.0 * epsilon);
            let dvy_dy = (at([x, y + epsilon])[1] - at([x, y - epsilon])[1]) / (2.0 * epsilon);
            let divergence = dvx_dx + dvy_dy;
            // The field peaks near 16 UV/s; a stream-function curl's numeric
            // divergence at this epsilon stays orders of magnitude below it.
            assert!(
                divergence.abs() < 0.05,
                "divergence {divergence} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn contour_flows_along_isolines_with_gradient_confidence() {
        let params = ProceduralFieldParams {
            scale: 0.0,
            rate: 0.0,
        };
        // A pure horizontal luma ramp: gradient +x, isolines vertical.
        let ramp = |uv: [f32; 2]| -> [f32; 4] { [uv[0], uv[0], uv[0], 1.0] };
        let step = [0.05_f32, 0.05];
        let sample = procedural_field_sample(
            ProceduralFieldKind::Contour,
            [0.5, 0.5],
            step,
            0.0,
            params,
            &ramp,
        );
        // luma(uv) = uv.x, central difference = step.x, tangent = (0, gx).
        let gx = step[0];
        let expected_y = PROCEDURAL_FIELD_MAX_SPEED * gx * 1.0;
        assert!(sample.velocity_uv_per_second[0].abs() < 1.0e-4);
        assert!((sample.velocity_uv_per_second[1] - expected_y).abs() < 1.0e-3);
        assert!((sample.confidence - (gx * 8.0)).abs() < 1.0e-4);
        // Flat content honestly contributes nothing.
        let flat = procedural_field_sample(
            ProceduralFieldKind::Contour,
            [0.5, 0.5],
            step,
            0.0,
            params,
            &|_uv| [0.25, 0.25, 0.25, 1.0],
        );
        assert_eq!(flat.velocity_uv_per_second, [0.0, 0.0]);
        assert_eq!(flat.confidence, 0.0);
    }

    #[test]
    fn chroma_steering_is_alpha_covered_so_hidden_rgb_steers_nothing() {
        let params = ProceduralFieldParams::default();
        // The image contract is covered premultiplied RGBA, so zero coverage
        // means zero channels — hostile hidden RGB cannot survive covering.
        let covered_hostile = |_uv: [f32; 2]| -> [f32; 4] { [0.0, 0.0, 0.0, 0.0] };
        let sample = procedural_field_sample(
            ProceduralFieldKind::Chroma,
            [0.4, 0.6],
            [0.01, 0.01],
            3.0,
            params,
            &covered_hostile,
        );
        assert_eq!(sample.velocity_uv_per_second, [0.0, 0.0]);
        assert_eq!(sample.confidence, 0.0);
        // A saturated red field steers along its YIQ chroma vector at rate 0.
        let red = procedural_field_sample(
            ProceduralFieldKind::Chroma,
            [0.4, 0.6],
            [0.01, 0.01],
            0.0,
            ProceduralFieldParams {
                scale: 0.5,
                rate: 0.0,
            },
            &|_uv| [1.0, 0.0, 0.0, 1.0],
        );
        let expected = [
            PROCEDURAL_FIELD_MAX_SPEED * 2.0 * 0.596,
            PROCEDURAL_FIELD_MAX_SPEED * 2.0 * 0.211,
        ];
        assert!((red.velocity_uv_per_second[0] - expected[0]).abs() < 1.0e-3);
        assert!((red.velocity_uv_per_second[1] - expected[1]).abs() < 1.0e-3);
        assert_eq!(red.visibility, 1.0);
    }

    #[test]
    fn procedural_velocity_always_lands_inside_the_canonical_range() {
        // A steep synthetic gradient through the frequency-scaled Contour
        // tangent exceeds the raw range and must clamp, not escape.
        let sample = procedural_field_sample(
            ProceduralFieldKind::Contour,
            [0.5, 0.5],
            [0.5, 0.5],
            0.0,
            ProceduralFieldParams {
                scale: 1.0,
                rate: 0.0,
            },
            &|uv| {
                let luma = if uv[0] > 0.5 { 1.0 } else { 0.0 };
                [luma, luma, luma, 1.0]
            },
        );
        for component in sample.velocity_uv_per_second {
            assert!(component.abs() <= MOTION_MAX_UV_PER_SECOND);
        }
        // Non-finite time takes the neutral zero rather than poisoning phase.
        let hostile_time = sample_pure(
            ProceduralFieldKind::Weave,
            [0.3, 0.3],
            f32::NAN,
            ProceduralFieldParams::default(),
        );
        assert!(hostile_time.velocity_uv_per_second[0].is_finite());
        assert!(hostile_time.velocity_uv_per_second[1].is_finite());
    }

    #[test]
    fn flow_shaping_sanitizes_to_neutral_values_and_zero_is_exact() {
        let hostile = FlowShapingParams {
            stretch: f32::NAN,
            edge_repel: -3.0,
            vector_trash: f32::INFINITY,
            trash_block_size: f32::NAN,
        }
        .sanitized();
        assert_eq!(hostile.stretch, 0.0);
        assert_eq!(hostile.edge_repel, 0.0);
        assert_eq!(hostile.vector_trash, 0.0);
        assert_eq!(hostile.trash_block_size, 16.0);
        assert!(hostile.is_exact_zero());
        assert!(FlowShapingParams::default().is_exact_zero());
        assert!(!FlowShapingParams {
            stretch: 0.1,
            ..FlowShapingParams::default()
        }
        .is_exact_zero());
        // A nonzero block size alone shapes nothing.
        assert!(FlowShapingParams {
            trash_block_size: 64.0,
            ..FlowShapingParams::default()
        }
        .is_exact_zero());
    }

    #[test]
    fn stretch_grows_radially_by_field_magnitude() {
        let params = FlowShapingParams {
            stretch: 0.5,
            ..FlowShapingParams::default()
        };
        // On the +x axis with a purely vertical velocity of magnitude 4, the
        // stretch term adds outward (+x) motion of 4 * 0.5.
        let shaped =
            shape_flow_velocity([0.0, 4.0], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0], 0.0, params);
        assert!((shaped[0] - 2.0).abs() < 1.0e-5);
        assert!((shaped[1] - 4.0).abs() < 1.0e-5);
        // At the centre there is no outward direction, and with a zero field
        // there is nothing to grow.
        assert_eq!(
            shape_flow_velocity([0.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], 0.0, params),
            [0.0, 0.0]
        );
    }

    #[test]
    fn edge_repel_pushes_down_the_saturated_luma_gradient() {
        let params = FlowShapingParams {
            edge_repel: 1.0,
            ..FlowShapingParams::default()
        };
        // A strong +x gradient saturates the push at one full luma step per
        // texel, so the contribution is exactly -PROCEDURAL_FIELD_MAX_SPEED.
        let shaped =
            shape_flow_velocity([0.0, 0.0], [0.4, 0.4], [0.5, 0.0], [0.0, 0.0], 0.0, params);
        assert!((shaped[0] + PROCEDURAL_FIELD_MAX_SPEED).abs() < 1.0e-5);
        assert_eq!(shaped[1], 0.0);
        // A weak gradient scales linearly below saturation.
        let weak =
            shape_flow_velocity([0.0, 0.0], [0.4, 0.4], [0.05, 0.0], [0.0, 0.0], 0.0, params);
        assert!((weak[0] + 0.05 * 8.0 * PROCEDURAL_FIELD_MAX_SPEED).abs() < 1.0e-4);
        // Flat content repels nothing.
        assert_eq!(
            shape_flow_velocity([1.0, 1.0], [0.4, 0.4], [0.0, 0.0], [0.0, 0.0], 0.0, params),
            [1.0, 1.0]
        );
    }

    #[test]
    fn vector_trash_fires_deterministically_by_cell_epoch_probability() {
        let params = FlowShapingParams {
            vector_trash: 1.0,
            trash_block_size: 16.0,
            ..FlowShapingParams::default()
        };
        // Probability one fires every cell; the shove is the deterministic
        // hash of (cell, epoch) and replays bit-identically.
        let first = shape_flow_velocity(
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [40.0, 24.0],
            1.0,
            params,
        );
        let replay = shape_flow_velocity(
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [40.0, 24.0],
            1.0,
            params,
        );
        assert_eq!(first, replay);
        assert!(first[0] != 0.0 || first[1] != 0.0);
        let expected_epoch = (1.0 * FLOW_TRASH_EVENT_HZ) as u32;
        let cell = [2, 1];
        let expected = [
            flow_trash_hash(cell, expected_epoch, 1).mul_add(2.0, -1.0)
                * 2.0
                * PROCEDURAL_FIELD_MAX_SPEED,
            flow_trash_hash(cell, expected_epoch, 2).mul_add(2.0, -1.0)
                * 2.0
                * PROCEDURAL_FIELD_MAX_SPEED,
        ];
        assert!((first[0] - expected[0]).abs() < 1.0e-5);
        assert!((first[1] - expected[1]).abs() < 1.0e-5);
        // Probability zero never fires, and the gate itself is the honest
        // per-cell hash: over many cells at 0.25 roughly a quarter fire.
        assert_eq!(
            shape_flow_velocity(
                [0.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.0],
                [40.0, 24.0],
                1.0,
                FlowShapingParams::default()
            ),
            [0.0, 0.0]
        );
        let quarter = FlowShapingParams {
            vector_trash: 0.25,
            trash_block_size: 2.0,
            ..FlowShapingParams::default()
        };
        let fired = (0..64)
            .filter(|index| {
                let px = [(index % 8) as f32 * 2.0, (index / 8) as f32 * 2.0];
                shape_flow_velocity([0.0, 0.0], [0.5, 0.5], [0.0, 0.0], px, 0.0, quarter)
                    != [0.0, 0.0]
            })
            .count();
        assert!((8..=24).contains(&fired), "fired {fired} of 64 cells");
    }

    #[test]
    fn shaped_velocity_always_lands_inside_the_canonical_range() {
        let params = FlowShapingParams {
            stretch: 1.0,
            edge_repel: 1.0,
            vector_trash: 1.0,
            trash_block_size: 2.0,
        };
        let shaped = shape_flow_velocity(
            [63.0, -63.0],
            [0.9, 0.1],
            [1.0, -1.0],
            [500.0, 300.0],
            2.5,
            params,
        );
        for component in shaped {
            assert!(component.abs() <= MOTION_MAX_UV_PER_SECOND);
            assert!(component.is_finite());
        }
        // Hostile inputs take neutral zeros rather than poisoning the sum.
        let hostile = shape_flow_velocity(
            [f32::NAN, 1.0],
            [0.5, 0.5],
            [f32::NAN, 0.0],
            [f32::NAN, 0.0],
            f32::NAN,
            params,
        );
        assert!(hostile[0].is_finite() && hostile[1].is_finite());
    }

    #[test]
    fn procedural_fields_charge_no_luma_bytes_in_preflight() {
        let request = |field_source: MotionFieldSource| MotionScopeResourceRequest {
            source_dimensions: [640, 360],
            output_dimensions: [640, 360],
            params: MotionParams {
                field_source,
                ..MotionParams::default()
            }
            .sanitized(),
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: true,
            required_as_garden_signal: false,
            required_as_study_input: false,
        };
        let procedural = MotionResourcePlan::preflight(
            &[request(MotionFieldSource::Procedural(
                ProceduralFieldKind::Curl,
            ))],
            generous_limits(),
        )
        .unwrap();
        assert_eq!(procedural.luma_bytes, 0);
        assert_eq!(procedural.active_field_slots, 1);
        assert!(procedural.vector_bytes > 0);
        let lattice = MotionResourcePlan::preflight(
            &[request(MotionFieldSource::Lattice)],
            generous_limits(),
        )
        .unwrap();
        assert!(lattice.luma_bytes > 0);
        // Same vector/gate surfaces either way: the procedural delta is one
        // low-resolution pass and zero bytes.
        assert_eq!(procedural.vector_bytes, lattice.vector_bytes);
        assert_eq!(procedural.gate_bytes, lattice.gate_bytes);
    }
}
