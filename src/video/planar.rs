//! Evidence-gated planar software-delivery prototype.
//!
//! This module deliberately does not replace [`super::DecodedVideoFrame`] or
//! enter the renderer. It freezes the smallest contract that can be measured
//! first: one immutable, bounded physical allocation can carry tightly packed
//! YUV420P, NV12, or P010 planes while frame/PTS/source-generation and codec
//! motion remain attached to the same owned frame object. The legacy packed
//! variant retains the exact existing [`DecodedImagePayload`] identity.
//!
//! The CPU conversion routine is an independent matrix/range/chroma-siting
//! reference. It is not an upload path and refuses HDR transfer functions;
//! integrating planar textures, tone mapping, pooling, or patch policy remains
//! behind the P4c measurement gate.

#![allow(
    dead_code,
    reason = "P4c is deliberately stopped at a measured-before-promotion contract; production consumers are forbidden until its GPU gate passes"
)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::media_safety::ABSOLUTE_MEDIA_MAX_EDGE;

use super::{
    BitDepth, ChromaLocation, CodecMotionProduct, DecodedImagePayload, DecodedVideoFrame,
    FrameMetadata, MatrixCoefficients, PixelFamily, SourceColorDescriptor, SourceColorRange,
    SourceFieldOrder, TransferCharacteristic,
};

/// One admitted planar frame may retain at most 128 MiB of physical plane
/// bytes. This admits 8K 4:2:0 at 8 or 10 bits while rejecting hostile 16K
/// material before allocation.
pub const MAX_PLANAR_FRAME_BYTES: usize = 128 * 1024 * 1024;
/// A prototype budget cannot be configured above this aggregate ceiling.
pub const MAX_PLANAR_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
/// The CPU oracle is bounded independently because RGBA is larger than 4:2:0.
pub const MAX_CPU_REFERENCE_RGBA_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_PLANAR_PLANES: usize = 3;

static NEXT_PLANAR_PAYLOAD_ID: AtomicU64 = AtomicU64::new(1);

/// The only planar software formats admitted by this prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanarPixelFormat {
    Yuv420p8,
    Nv12,
    /// Little-endian 16-bit words with the ten meaningful bits in bits 15..6.
    P010Le,
}

impl PlanarPixelFormat {
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Yuv420p8 | Self::Nv12 => 8,
            Self::P010Le => 10,
        }
    }

    pub const fn plane_count(self) -> usize {
        match self {
            Self::Yuv420p8 => 3,
            Self::Nv12 | Self::P010Le => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PlanarPlaneKind {
    #[default]
    Y,
    U,
    V,
    Uv,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanarPlaneLayout {
    pub kind: PlanarPlaneKind,
    pub width: u32,
    pub height: u32,
    /// Bytes containing meaningful samples in one tightly stored row.
    pub row_bytes: usize,
    /// Always equal to `row_bytes` in the immutable prototype allocation.
    pub stride: usize,
    pub offset: usize,
    pub byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanarImageLayout {
    pub format: PlanarPixelFormat,
    pub width: u32,
    pub height: u32,
    planes: [PlanarPlaneLayout; MAX_PLANAR_PLANES],
    plane_count: u8,
    byte_len: usize,
}

impl PlanarImageLayout {
    pub fn new(
        format: PlanarPixelFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, PlanarImageError> {
        if width == 0 || height == 0 {
            return Err(PlanarImageError::ZeroDimensions);
        }
        if width > ABSOLUTE_MEDIA_MAX_EDGE || height > ABSOLUTE_MEDIA_MAX_EDGE {
            return Err(PlanarImageError::EdgeCap {
                width,
                height,
                max: ABSOLUTE_MEDIA_MAX_EDGE,
            });
        }

        let width_usize = usize::try_from(width).map_err(|_| PlanarImageError::Arithmetic)?;
        let height_usize = usize::try_from(height).map_err(|_| PlanarImageError::Arithmetic)?;
        let chroma_width = ceil_half(width_usize);
        let chroma_height = ceil_half(height_usize);
        let mut planes = [PlanarPlaneLayout::default(); MAX_PLANAR_PLANES];
        let mut offset = 0usize;

        let mut push_plane = |index: usize,
                              kind: PlanarPlaneKind,
                              plane_width: usize,
                              plane_height: usize,
                              row_bytes: usize|
         -> Result<(), PlanarImageError> {
            let byte_len = row_bytes
                .checked_mul(plane_height)
                .ok_or(PlanarImageError::Arithmetic)?;
            let next = offset
                .checked_add(byte_len)
                .ok_or(PlanarImageError::Arithmetic)?;
            planes[index] = PlanarPlaneLayout {
                kind,
                width: u32::try_from(plane_width).map_err(|_| PlanarImageError::Arithmetic)?,
                height: u32::try_from(plane_height).map_err(|_| PlanarImageError::Arithmetic)?,
                row_bytes,
                stride: row_bytes,
                offset,
                byte_len,
            };
            offset = next;
            Ok(())
        };

        match format {
            PlanarPixelFormat::Yuv420p8 => {
                push_plane(
                    0,
                    PlanarPlaneKind::Y,
                    width_usize,
                    height_usize,
                    width_usize,
                )?;
                push_plane(
                    1,
                    PlanarPlaneKind::U,
                    chroma_width,
                    chroma_height,
                    chroma_width,
                )?;
                push_plane(
                    2,
                    PlanarPlaneKind::V,
                    chroma_width,
                    chroma_height,
                    chroma_width,
                )?;
            }
            PlanarPixelFormat::Nv12 => {
                push_plane(
                    0,
                    PlanarPlaneKind::Y,
                    width_usize,
                    height_usize,
                    width_usize,
                )?;
                push_plane(
                    1,
                    PlanarPlaneKind::Uv,
                    chroma_width,
                    chroma_height,
                    chroma_width
                        .checked_mul(2)
                        .ok_or(PlanarImageError::Arithmetic)?,
                )?;
            }
            PlanarPixelFormat::P010Le => {
                push_plane(
                    0,
                    PlanarPlaneKind::Y,
                    width_usize,
                    height_usize,
                    width_usize
                        .checked_mul(2)
                        .ok_or(PlanarImageError::Arithmetic)?,
                )?;
                push_plane(
                    1,
                    PlanarPlaneKind::Uv,
                    chroma_width,
                    chroma_height,
                    chroma_width
                        .checked_mul(4)
                        .ok_or(PlanarImageError::Arithmetic)?,
                )?;
            }
        }

        if offset > MAX_PLANAR_FRAME_BYTES {
            return Err(PlanarImageError::FrameByteCap {
                requested: offset,
                max: MAX_PLANAR_FRAME_BYTES,
            });
        }
        Ok(Self {
            format,
            width,
            height,
            planes,
            plane_count: format.plane_count() as u8,
            byte_len: offset,
        })
    }

    pub const fn plane_count(self) -> usize {
        self.plane_count as usize
    }

    pub fn planes(&self) -> &[PlanarPlaneLayout] {
        &self.planes[..self.plane_count()]
    }

    pub fn plane(&self, kind: PlanarPlaneKind) -> Option<PlanarPlaneLayout> {
        self.planes()
            .iter()
            .copied()
            .find(|plane| plane.kind == kind)
    }

    pub const fn byte_len(self) -> usize {
        self.byte_len
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarPlaneInput<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}

impl<'a> PlanarPlaneInput<'a> {
    pub const fn new(data: &'a [u8], stride: usize) -> Self {
        Self { data, stride }
    }
}

/// A fixed-shape input vocabulary prevents an attacker-controlled plane list.
#[derive(Debug, Clone, Copy)]
pub enum PlanarPlaneInputs<'a> {
    Yuv420p8 {
        y: PlanarPlaneInput<'a>,
        u: PlanarPlaneInput<'a>,
        v: PlanarPlaneInput<'a>,
    },
    Nv12 {
        y: PlanarPlaneInput<'a>,
        uv: PlanarPlaneInput<'a>,
    },
    P010Le {
        y: PlanarPlaneInput<'a>,
        uv: PlanarPlaneInput<'a>,
    },
}

/// Aggregate physical-byte ceiling shared by any prototype planar payloads.
#[derive(Debug)]
pub struct PlanarAllocationBudget {
    max_bytes: u64,
    live_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    allocations: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanarAllocationSnapshot {
    pub max_bytes: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
    pub allocations: u64,
}

impl PlanarAllocationBudget {
    pub fn new(max_bytes: u64) -> Result<Arc<Self>, PlanarImageError> {
        if max_bytes == 0 || max_bytes > MAX_PLANAR_BUDGET_BYTES {
            return Err(PlanarImageError::BudgetConfiguration {
                requested: max_bytes,
                max: MAX_PLANAR_BUDGET_BYTES,
            });
        }
        Ok(Arc::new(Self {
            max_bytes,
            live_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
        }))
    }

    pub fn snapshot(&self) -> PlanarAllocationSnapshot {
        PlanarAllocationSnapshot {
            max_bytes: self.max_bytes,
            live_bytes: self.live_bytes.load(Ordering::Acquire),
            peak_bytes: self.peak_bytes.load(Ordering::Acquire),
            allocations: self.allocations.load(Ordering::Acquire),
        }
    }

    fn try_reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<PlanarAllocationLease, PlanarImageError> {
        let mut observed = self.live_bytes.load(Ordering::Acquire);
        loop {
            let requested = observed
                .checked_add(bytes)
                .ok_or(PlanarImageError::Arithmetic)?;
            if requested > self.max_bytes {
                return Err(PlanarImageError::AggregateByteCap {
                    live: observed,
                    requested: bytes,
                    max: self.max_bytes,
                });
            }
            match self.live_bytes.compare_exchange_weak(
                observed,
                requested,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak_bytes.fetch_max(requested, Ordering::AcqRel);
                    return Ok(PlanarAllocationLease {
                        budget: self.clone(),
                        bytes,
                    });
                }
                Err(actual) => observed = actual,
            }
        }
    }
}

#[derive(Debug)]
struct PlanarAllocationLease {
    budget: Arc<PlanarAllocationBudget>,
    bytes: u64,
}

impl Drop for PlanarAllocationLease {
    fn drop(&mut self) {
        self.budget
            .live_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct PlanarImagePayloadInner {
    identity: u64,
    layout: PlanarImageLayout,
    /// A single allocation owns every tightly packed plane. It is immutable
    /// after construction and therefore safe to share with cache/upload roles.
    bytes: Vec<u8>,
    _lease: PlanarAllocationLease,
}

/// Immutable Arc-owned planar pixels. Clones share one identity/allocation and
/// do not reserve aggregate bytes again.
#[derive(Clone)]
pub struct PlanarImagePayload {
    inner: Arc<PlanarImagePayloadInner>,
}

impl PlanarImagePayload {
    pub fn from_planes(
        layout: PlanarImageLayout,
        inputs: PlanarPlaneInputs<'_>,
        budget: &Arc<PlanarAllocationBudget>,
    ) -> Result<Self, PlanarImageError> {
        let fixed_inputs = match (layout.format, inputs) {
            (PlanarPixelFormat::Yuv420p8, PlanarPlaneInputs::Yuv420p8 { y, u, v }) => {
                [Some(y), Some(u), Some(v)]
            }
            (PlanarPixelFormat::Nv12, PlanarPlaneInputs::Nv12 { y, uv })
            | (PlanarPixelFormat::P010Le, PlanarPlaneInputs::P010Le { y, uv }) => {
                [Some(y), Some(uv), None]
            }
            _ => return Err(PlanarImageError::InputFormatMismatch),
        };

        // Validate all borrowed spans before reserving aggregate bytes or
        // allocating the destination.
        for (plane, input) in layout.planes().iter().zip(fixed_inputs.iter()) {
            validate_plane_input(*plane, input.ok_or(PlanarImageError::PlaneCount)?)?;
        }

        let physical_bytes =
            u64::try_from(layout.byte_len()).map_err(|_| PlanarImageError::Arithmetic)?;
        let lease = budget.try_reserve(physical_bytes)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(layout.byte_len())
            .map_err(|_| PlanarImageError::Allocation)?;
        for (plane, input) in layout.planes().iter().zip(fixed_inputs.iter()) {
            let input = input.ok_or(PlanarImageError::PlaneCount)?;
            for row in 0..usize::try_from(plane.height).map_err(|_| PlanarImageError::Arithmetic)? {
                let start = row
                    .checked_mul(input.stride)
                    .ok_or(PlanarImageError::Arithmetic)?;
                let end = start
                    .checked_add(plane.row_bytes)
                    .ok_or(PlanarImageError::Arithmetic)?;
                bytes.extend_from_slice(&input.data[start..end]);
            }
        }
        debug_assert_eq!(bytes.len(), layout.byte_len());
        budget.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            inner: Arc::new(PlanarImagePayloadInner {
                identity: NEXT_PLANAR_PAYLOAD_ID
                    .fetch_add(1, Ordering::Relaxed)
                    .max(1),
                layout,
                bytes,
                _lease: lease,
            }),
        })
    }

    pub fn identity(&self) -> u64 {
        self.inner.identity
    }

    pub fn layout(&self) -> PlanarImageLayout {
        self.inner.layout
    }

    pub fn byte_len(&self) -> usize {
        self.inner.bytes.len()
    }

    pub fn plane(&self, kind: PlanarPlaneKind) -> Option<PlanarPlane<'_>> {
        let layout = self.inner.layout.plane(kind)?;
        let end = layout.offset.checked_add(layout.byte_len)?;
        Some(PlanarPlane {
            layout,
            data: self.inner.bytes.get(layout.offset..end)?,
        })
    }

    /// Independent CPU matrix/range/chroma-siting oracle. The returned RGBA
    /// remains source-encoded nonlinear RGB; it performs no gamut conversion,
    /// transfer linearization, or tone map and records that law explicitly.
    pub fn to_rgba8_cpu_reference(
        &self,
        color: SourceColorDescriptor,
    ) -> Result<CpuPlanarConversion, PlanarConversionError> {
        let contract = CpuConversionContract::from_descriptor(self.layout().format, color)?;
        let pixel_count = usize::try_from(self.layout().width)
            .map_err(|_| PlanarConversionError::Arithmetic)?
            .checked_mul(
                usize::try_from(self.layout().height)
                    .map_err(|_| PlanarConversionError::Arithmetic)?,
            )
            .ok_or(PlanarConversionError::Arithmetic)?;
        let output_len = pixel_count
            .checked_mul(4)
            .ok_or(PlanarConversionError::Arithmetic)?;
        if output_len > MAX_CPU_REFERENCE_RGBA_BYTES {
            return Err(PlanarConversionError::OutputByteCap {
                requested: output_len,
                max: MAX_CPU_REFERENCE_RGBA_BYTES,
            });
        }
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(output_len)
            .map_err(|_| PlanarConversionError::Allocation)?;
        let width =
            usize::try_from(self.layout().width).map_err(|_| PlanarConversionError::Arithmetic)?;
        let height =
            usize::try_from(self.layout().height).map_err(|_| PlanarConversionError::Arithmetic)?;
        for y in 0..height {
            for x in 0..width {
                let luma = self.luma_code(x, y)?;
                let (chroma_u, chroma_v) =
                    self.chroma_code(x, y, contract.chroma_location, contract.bit_depth)?;
                rgba.extend_from_slice(&contract.convert(luma, chroma_u, chroma_v));
            }
        }
        debug_assert_eq!(rgba.len(), output_len);
        Ok(CpuPlanarConversion {
            rgba,
            policy: CpuPlanarConversionPolicy {
                law: CpuPlanarConversionLaw::SourceEncodedSdrRgba8NoGamutMapping,
                format: self.layout().format,
                bit_depth: contract.bit_depth,
                range: contract.range,
                matrix: contract.matrix,
                transfer: contract.transfer,
                chroma_location: contract.chroma_location,
            },
        })
    }

    fn luma_code(&self, x: usize, y: usize) -> Result<f64, PlanarConversionError> {
        let plane = self
            .plane(PlanarPlaneKind::Y)
            .ok_or(PlanarConversionError::MissingPlane)?;
        match self.layout().format {
            PlanarPixelFormat::Yuv420p8 | PlanarPixelFormat::Nv12 => plane
                .sample_u8(x, y)
                .map(f64::from)
                .ok_or(PlanarConversionError::PlaneBounds),
            PlanarPixelFormat::P010Le => plane
                .sample_p010_luma(x, y)
                .map(f64::from)
                .ok_or(PlanarConversionError::PlaneBounds),
        }
    }

    fn chroma_code(
        &self,
        x: usize,
        y: usize,
        location: ChromaLocation,
        bit_depth: u8,
    ) -> Result<(f64, f64), PlanarConversionError> {
        let (horizontal_offset, vertical_offset) = chroma_sample_offset(location)?;
        let chroma_x = (x as f64 - horizontal_offset) / 2.0;
        let chroma_y = (y as f64 - vertical_offset) / 2.0;
        let max_code = ((1u32 << bit_depth) - 1) as f64;
        let read = |component: usize, sx: usize, sy: usize| -> Result<f64, PlanarConversionError> {
            match self.layout().format {
                PlanarPixelFormat::Yuv420p8 => {
                    let kind = if component == 0 {
                        PlanarPlaneKind::U
                    } else {
                        PlanarPlaneKind::V
                    };
                    self.plane(kind)
                        .and_then(|plane| plane.sample_u8(sx, sy))
                        .map(f64::from)
                        .ok_or(PlanarConversionError::PlaneBounds)
                }
                PlanarPixelFormat::Nv12 => self
                    .plane(PlanarPlaneKind::Uv)
                    .and_then(|plane| plane.sample_u8(sx * 2 + component, sy))
                    .map(f64::from)
                    .ok_or(PlanarConversionError::PlaneBounds),
                PlanarPixelFormat::P010Le => self
                    .plane(PlanarPlaneKind::Uv)
                    .and_then(|plane| plane.sample_p010(sx, sy, component))
                    .map(f64::from)
                    .ok_or(PlanarConversionError::PlaneBounds),
            }
        };

        let plane = match self.layout().format {
            PlanarPixelFormat::Yuv420p8 => self
                .plane(PlanarPlaneKind::U)
                .ok_or(PlanarConversionError::MissingPlane)?,
            PlanarPixelFormat::Nv12 | PlanarPixelFormat::P010Le => self
                .plane(PlanarPlaneKind::Uv)
                .ok_or(PlanarConversionError::MissingPlane)?,
        };
        let plane_width =
            usize::try_from(plane.layout.width).map_err(|_| PlanarConversionError::Arithmetic)?;
        let plane_height =
            usize::try_from(plane.layout.height).map_err(|_| PlanarConversionError::Arithmetic)?;
        let sample_component = |component| {
            bilinear_sample(chroma_x, chroma_y, plane_width, plane_height, |sx, sy| {
                read(component, sx, sy)
            })
        };
        let u = sample_component(0)?.clamp(0.0, max_code);
        let v = sample_component(1)?.clamp(0.0, max_code);
        Ok((u, v))
    }
}

impl fmt::Debug for PlanarImagePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlanarImagePayload")
            .field("identity", &self.identity())
            .field("layout", &self.layout())
            .field("bytes", &self.byte_len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarPlane<'a> {
    pub layout: PlanarPlaneLayout,
    pub data: &'a [u8],
}

impl PlanarPlane<'_> {
    fn sample_u8(self, x: usize, y: usize) -> Option<u8> {
        let offset = y.checked_mul(self.layout.stride)?.checked_add(x)?;
        self.data.get(offset).copied()
    }

    fn sample_p010_luma(self, x: usize, y: usize) -> Option<u16> {
        let offset = y
            .checked_mul(self.layout.stride)?
            .checked_add(x.checked_mul(2)?)?;
        let low = *self.data.get(offset)?;
        let high = *self.data.get(offset + 1)?;
        Some(u16::from_le_bytes([low, high]) >> 6)
    }

    fn sample_p010(self, x: usize, y: usize, component: usize) -> Option<u16> {
        let component_offset = x.checked_mul(4)?.checked_add(component.checked_mul(2)?)?;
        let offset = y
            .checked_mul(self.layout.stride)?
            .checked_add(component_offset)?;
        let low = *self.data.get(offset)?;
        let high = *self.data.get(offset + 1)?;
        Some(u16::from_le_bytes([low, high]) >> 6)
    }
}

/// Additive policy vocabulary. `LegacyRgba` is intentionally the serde and
/// Rust default; merely compiling this prototype cannot change an old patch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanarDeliveryPolicy {
    #[default]
    LegacyRgba,
    MetadataManaged,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanarDeliverySettings {
    pub policy: PlanarDeliveryPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarFallbackReason {
    LegacyPolicy,
    Interlaced,
    IncompleteMetadata,
    DescriptorMismatch,
    UnsupportedMatrix,
    UnsupportedTransfer,
    HdrToneMapRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarDeliveryDecision {
    PackedRgbaFallback(PlanarFallbackReason),
    /// Contract-only admission. No production decoder or renderer consumes
    /// this result until transport and end-to-end P4c gates pass.
    PrototypePlanar(PlanarPixelFormat),
}

pub fn prototype_delivery_decision(
    policy: PlanarDeliveryPolicy,
    format: PlanarPixelFormat,
    color: SourceColorDescriptor,
    field_order: SourceFieldOrder,
) -> PlanarDeliveryDecision {
    if policy == PlanarDeliveryPolicy::LegacyRgba {
        return PlanarDeliveryDecision::PackedRgbaFallback(PlanarFallbackReason::LegacyPolicy);
    }
    if !matches!(field_order, SourceFieldOrder::Progressive) {
        return PlanarDeliveryDecision::PackedRgbaFallback(
            if matches!(field_order, SourceFieldOrder::Unspecified) {
                PlanarFallbackReason::IncompleteMetadata
            } else {
                PlanarFallbackReason::Interlaced
            },
        );
    }
    if color.pixel_family.value != PixelFamily::Yuv
        || color.bit_depth.value != BitDepth::Bits(format.bit_depth())
    {
        return PlanarDeliveryDecision::PackedRgbaFallback(
            PlanarFallbackReason::DescriptorMismatch,
        );
    }
    if matches!(color.range.value, SourceColorRange::Unspecified)
        || matches!(color.matrix.value, MatrixCoefficients::Unspecified)
        || matches!(color.transfer.value, TransferCharacteristic::Unspecified)
        || matches!(color.chroma_location.value, ChromaLocation::Unspecified)
    {
        return PlanarDeliveryDecision::PackedRgbaFallback(
            PlanarFallbackReason::IncompleteMetadata,
        );
    }
    if matches!(
        color.transfer.value,
        TransferCharacteristic::Pq | TransferCharacteristic::Hlg
    ) {
        return PlanarDeliveryDecision::PackedRgbaFallback(
            PlanarFallbackReason::HdrToneMapRequired,
        );
    }
    if !supported_sdr_transfer(color.transfer.value) {
        return PlanarDeliveryDecision::PackedRgbaFallback(
            PlanarFallbackReason::UnsupportedTransfer,
        );
    }
    if matrix_kr_kb(color.matrix.value).is_none() {
        return PlanarDeliveryDecision::PackedRgbaFallback(PlanarFallbackReason::UnsupportedMatrix);
    }
    PlanarDeliveryDecision::PrototypePlanar(format)
}

/// Transitional owned image vocabulary. It is intentionally separate from
/// the production decoded-frame type until GPU evidence exists.
#[derive(Debug, Clone)]
pub enum DecodedImageDelivery {
    PackedRgba8(DecodedImagePayload),
    Planar(PlanarImagePayload),
}

impl DecodedImageDelivery {
    pub fn identity(&self) -> u64 {
        match self {
            Self::PackedRgba8(payload) => payload.identity(),
            Self::Planar(payload) => payload.identity(),
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Self::PackedRgba8(payload) => payload.len(),
            Self::Planar(payload) => payload.byte_len(),
        }
    }

    pub fn legacy_rgba_bytes(&self) -> Result<&[u8], PlanarImageError> {
        match self {
            Self::PackedRgba8(payload) => Ok(payload.as_slice()),
            Self::Planar(_) => Err(PlanarImageError::ExplicitConversionRequired),
        }
    }
}

/// Image plus the exact existing metadata/motion objects. Wrapping or
/// unwrapping a packed frame moves these fields; it never retags or clones
/// pixel bytes.
#[derive(Debug, Clone)]
pub struct DecodedDeliveryFrame {
    pub image: DecodedImageDelivery,
    pub metadata: FrameMetadata,
    pub codec_motion: Option<CodecMotionProduct>,
}

impl DecodedDeliveryFrame {
    pub fn from_legacy(frame: DecodedVideoFrame) -> Self {
        Self {
            image: DecodedImageDelivery::PackedRgba8(frame.rgba),
            metadata: frame.metadata,
            codec_motion: frame.codec_motion,
        }
    }

    #[allow(
        clippy::result_large_err,
        reason = "the stopped prototype returns the original frame allocation on an explicit planar refusal; boxing would alter its ownership/identity contract"
    )]
    pub fn into_legacy(self) -> Result<DecodedVideoFrame, Self> {
        let Self {
            image,
            metadata,
            codec_motion,
        } = self;
        match image {
            DecodedImageDelivery::PackedRgba8(rgba) => Ok(DecodedVideoFrame {
                rgba,
                metadata,
                codec_motion,
            }),
            image @ DecodedImageDelivery::Planar(_) => Err(Self {
                image,
                metadata,
                codec_motion,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPlanarConversionLaw {
    /// Matrix/range/chroma reconstruction only. Output is nonlinear source RGB
    /// code values with no transfer/gamut/tone-map claim.
    SourceEncodedSdrRgba8NoGamutMapping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuPlanarConversionPolicy {
    pub law: CpuPlanarConversionLaw,
    pub format: PlanarPixelFormat,
    pub bit_depth: u8,
    pub range: SourceColorRange,
    pub matrix: MatrixCoefficients,
    pub transfer: TransferCharacteristic,
    pub chroma_location: ChromaLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuPlanarConversion {
    pub rgba: Vec<u8>,
    pub policy: CpuPlanarConversionPolicy,
}

struct CpuConversionContract {
    bit_depth: u8,
    range: SourceColorRange,
    matrix: MatrixCoefficients,
    transfer: TransferCharacteristic,
    chroma_location: ChromaLocation,
    kr: f64,
    kb: f64,
}

impl CpuConversionContract {
    fn from_descriptor(
        format: PlanarPixelFormat,
        color: SourceColorDescriptor,
    ) -> Result<Self, PlanarConversionError> {
        if color.pixel_family.value != PixelFamily::Yuv
            || color.bit_depth.value != BitDepth::Bits(format.bit_depth())
        {
            return Err(PlanarConversionError::DescriptorMismatch);
        }
        let range = match color.range.value {
            value @ (SourceColorRange::Limited | SourceColorRange::Full) => value,
            SourceColorRange::Unspecified => return Err(PlanarConversionError::UnspecifiedRange),
        };
        let matrix = color.matrix.value;
        let (kr, kb) =
            matrix_kr_kb(matrix).ok_or(PlanarConversionError::UnsupportedMatrix(matrix))?;
        let transfer = color.transfer.value;
        if matches!(
            transfer,
            TransferCharacteristic::Pq | TransferCharacteristic::Hlg
        ) {
            return Err(PlanarConversionError::HdrToneMapRequired(transfer));
        }
        if matches!(transfer, TransferCharacteristic::Unspecified) {
            return Err(PlanarConversionError::UnspecifiedTransfer);
        }
        if !supported_sdr_transfer(transfer) {
            return Err(PlanarConversionError::UnsupportedTransfer(transfer));
        }
        let chroma_location = color.chroma_location.value;
        if matches!(chroma_location, ChromaLocation::Unspecified) {
            return Err(PlanarConversionError::UnspecifiedChromaLocation);
        }
        Ok(Self {
            bit_depth: format.bit_depth(),
            range,
            matrix,
            transfer,
            chroma_location,
            kr,
            kb,
        })
    }

    fn convert(&self, y_code: f64, u_code: f64, v_code: f64) -> [u8; 4] {
        let scale = (1u32 << self.bit_depth.saturating_sub(8)) as f64;
        let max = ((1u32 << self.bit_depth) - 1) as f64;
        let (y, cb, cr) = match self.range {
            SourceColorRange::Limited => (
                (y_code - 16.0 * scale) / (219.0 * scale),
                (u_code - 128.0 * scale) / (224.0 * scale),
                (v_code - 128.0 * scale) / (224.0 * scale),
            ),
            SourceColorRange::Full => (
                y_code / max,
                (u_code - (1u32 << (self.bit_depth - 1)) as f64) / max,
                (v_code - (1u32 << (self.bit_depth - 1)) as f64) / max,
            ),
            SourceColorRange::Unspecified => unreachable!("validated conversion range"),
        };
        let kg = 1.0 - self.kr - self.kb;
        let red = y + (2.0 - 2.0 * self.kr) * cr;
        let blue = y + (2.0 - 2.0 * self.kb) * cb;
        let green = y
            - self.kb * (2.0 - 2.0 * self.kb) / kg * cb
            - self.kr * (2.0 - 2.0 * self.kr) / kg * cr;
        [quantize(red), quantize(green), quantize(blue), 255]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanarImageError {
    ZeroDimensions,
    EdgeCap {
        width: u32,
        height: u32,
        max: u32,
    },
    FrameByteCap {
        requested: usize,
        max: usize,
    },
    BudgetConfiguration {
        requested: u64,
        max: u64,
    },
    AggregateByteCap {
        live: u64,
        requested: u64,
        max: u64,
    },
    InputFormatMismatch,
    PlaneCount,
    PlaneStride {
        kind: PlanarPlaneKind,
        stride: usize,
        row_bytes: usize,
    },
    PlaneData {
        kind: PlanarPlaneKind,
        required: usize,
        available: usize,
    },
    ExplicitConversionRequired,
    Arithmetic,
    Allocation,
}

impl fmt::Display for PlanarImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PlanarImageError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanarConversionError {
    DescriptorMismatch,
    UnspecifiedRange,
    UnsupportedMatrix(MatrixCoefficients),
    UnspecifiedTransfer,
    UnsupportedTransfer(TransferCharacteristic),
    HdrToneMapRequired(TransferCharacteristic),
    UnspecifiedChromaLocation,
    MissingPlane,
    PlaneBounds,
    OutputByteCap { requested: usize, max: usize },
    Arithmetic,
    Allocation,
}

impl fmt::Display for PlanarConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PlanarConversionError {}

fn validate_plane_input(
    plane: PlanarPlaneLayout,
    input: PlanarPlaneInput<'_>,
) -> Result<(), PlanarImageError> {
    if input.stride < plane.row_bytes {
        return Err(PlanarImageError::PlaneStride {
            kind: plane.kind,
            stride: input.stride,
            row_bytes: plane.row_bytes,
        });
    }
    let height = usize::try_from(plane.height).map_err(|_| PlanarImageError::Arithmetic)?;
    let required = height
        .saturating_sub(1)
        .checked_mul(input.stride)
        .and_then(|prefix| prefix.checked_add(plane.row_bytes))
        .ok_or(PlanarImageError::Arithmetic)?;
    if input.data.len() < required {
        return Err(PlanarImageError::PlaneData {
            kind: plane.kind,
            required,
            available: input.data.len(),
        });
    }
    Ok(())
}

fn ceil_half(value: usize) -> usize {
    value / 2 + value % 2
}

fn matrix_kr_kb(matrix: MatrixCoefficients) -> Option<(f64, f64)> {
    match matrix {
        MatrixCoefficients::Bt709 => Some((0.2126, 0.0722)),
        MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170M => Some((0.299, 0.114)),
        MatrixCoefficients::Bt2020Ncl => Some((0.2627, 0.0593)),
        _ => None,
    }
}

fn supported_sdr_transfer(transfer: TransferCharacteristic) -> bool {
    matches!(
        transfer,
        TransferCharacteristic::Bt709
            | TransferCharacteristic::Gamma22
            | TransferCharacteristic::Gamma28
            | TransferCharacteristic::Smpte170M
            | TransferCharacteristic::Smpte240M
            | TransferCharacteristic::Srgb
            | TransferCharacteristic::Bt2020_10
            | TransferCharacteristic::Bt2020_12
    )
}

fn chroma_sample_offset(location: ChromaLocation) -> Result<(f64, f64), PlanarConversionError> {
    match location {
        ChromaLocation::Left => Ok((0.0, 0.5)),
        ChromaLocation::Center => Ok((0.5, 0.5)),
        ChromaLocation::TopLeft => Ok((0.0, 0.0)),
        ChromaLocation::Top => Ok((0.5, 0.0)),
        ChromaLocation::BottomLeft => Ok((0.0, 1.0)),
        ChromaLocation::Bottom => Ok((0.5, 1.0)),
        ChromaLocation::Unspecified => Err(PlanarConversionError::UnspecifiedChromaLocation),
    }
}

fn bilinear_sample<F>(
    x: f64,
    y: f64,
    width: usize,
    height: usize,
    mut read: F,
) -> Result<f64, PlanarConversionError>
where
    F: FnMut(usize, usize) -> Result<f64, PlanarConversionError>,
{
    if width == 0 || height == 0 {
        return Err(PlanarConversionError::PlaneBounds);
    }
    let x = x.clamp(0.0, width.saturating_sub(1) as f64);
    let y = y.clamp(0.0, height.saturating_sub(1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0.saturating_add(1).min(width - 1);
    let y1 = y0.saturating_add(1).min(height - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let top = read(x0, y0)? * (1.0 - tx) + read(x1, y0)? * tx;
    let bottom = read(x0, y1)? * (1.0 - tx) + read(x1, y1)? * tx;
    Ok(top * (1.0 - ty) + bottom * ty)
}

fn quantize(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MOTION_ALGORITHM_VERSION;
    use crate::video::{
        CodecFrameIdentity, CodecMotionFrame, CodecMotionFrameType, CodecMotionProvenance,
        CodecMotionStatus, DescriptorProvenance, DescriptorValue,
    };

    fn color(
        bits: u8,
        range: SourceColorRange,
        matrix: MatrixCoefficients,
        transfer: TransferCharacteristic,
        location: ChromaLocation,
    ) -> SourceColorDescriptor {
        let declared = DescriptorProvenance::CodecDeclared;
        SourceColorDescriptor {
            pixel_family: DescriptorValue::new(PixelFamily::Yuv, declared),
            bit_depth: DescriptorValue::new(BitDepth::Bits(bits), declared),
            range: DescriptorValue::new(range, declared),
            matrix: DescriptorValue::new(matrix, declared),
            transfer: DescriptorValue::new(transfer, declared),
            chroma_location: DescriptorValue::new(location, declared),
            chroma_subsampling: DescriptorValue::new(
                super::super::ChromaSubsampling {
                    horizontal_log2: 1,
                    vertical_log2: 1,
                },
                DescriptorProvenance::PixelFormatDerived,
            ),
            ..Default::default()
        }
    }

    fn yuv420_payload(
        width: u32,
        height: u32,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        budget: &Arc<PlanarAllocationBudget>,
    ) -> PlanarImagePayload {
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, width, height).unwrap();
        PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Yuv420p8 {
                y: PlanarPlaneInput::new(y, width as usize),
                u: PlanarPlaneInput::new(u, ceil_half(width as usize)),
                v: PlanarPlaneInput::new(v, ceil_half(width as usize)),
            },
            budget,
        )
        .unwrap()
    }

    fn p010_words(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| (value << 6).to_le_bytes())
            .collect()
    }

    fn motion_fixture(source_generation: u64, frame_ordinal: u64) -> CodecMotionProduct {
        CodecMotionFrame {
            source_dimensions: [2, 2],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation,
            frame_ordinal,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            provenance: CodecMotionProvenance::FfmpegExportMvs,
            frame_type: CodecMotionFrameType::Intra,
            status: CodecMotionStatus::Intra,
            past_reference_proof: None,
            vectors: Vec::new(),
        }
        .into()
    }

    #[test]
    fn layouts_are_exact_bounded_and_reject_the_first_byte_over_budget() {
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, 4, 2).unwrap();
        assert_eq!(layout.byte_len(), 12);
        assert_eq!(layout.plane_count(), 3);
        assert_eq!(layout.plane(PlanarPlaneKind::Y).unwrap().byte_len, 8);
        assert_eq!(layout.plane(PlanarPlaneKind::U).unwrap().byte_len, 2);
        assert_eq!(layout.plane(PlanarPlaneKind::V).unwrap().byte_len, 2);

        let under = PlanarAllocationBudget::new(11).unwrap();
        let error = PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Yuv420p8 {
                y: PlanarPlaneInput::new(&[16; 8], 4),
                u: PlanarPlaneInput::new(&[128; 2], 2),
                v: PlanarPlaneInput::new(&[128; 2], 2),
            },
            &under,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PlanarImageError::AggregateByteCap {
                live: 0,
                requested: 12,
                max: 11,
            }
        );
        assert_eq!(under.snapshot().live_bytes, 0);
        assert_eq!(under.snapshot().allocations, 0);

        assert!(matches!(
            PlanarImageLayout::new(
                PlanarPixelFormat::Yuv420p8,
                ABSOLUTE_MEDIA_MAX_EDGE,
                ABSOLUTE_MEDIA_MAX_EDGE,
            ),
            Err(PlanarImageError::FrameByteCap { .. })
        ));
        assert!(matches!(
            PlanarImageLayout::new(PlanarPixelFormat::Nv12, ABSOLUTE_MEDIA_MAX_EDGE + 1, 2,),
            Err(PlanarImageError::EdgeCap { .. })
        ));
    }

    #[test]
    fn one_arc_allocation_is_charged_once_and_padding_is_not_retained() {
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Nv12, 4, 2).unwrap();
        let budget = PlanarAllocationBudget::new(layout.byte_len() as u64).unwrap();
        let y = [16, 32, 64, 128, 99, 99, 235, 200, 100, 50];
        let uv = [128, 128, 90, 240, 77, 77];
        let payload = PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Nv12 {
                y: PlanarPlaneInput::new(&y, 6),
                uv: PlanarPlaneInput::new(&uv, 6),
            },
            &budget,
        )
        .unwrap();
        assert_eq!(
            payload.plane(PlanarPlaneKind::Y).unwrap().data,
            &[16, 32, 64, 128, 235, 200, 100, 50]
        );
        assert_eq!(
            payload.plane(PlanarPlaneKind::Uv).unwrap().data,
            &[128, 128, 90, 240]
        );
        let clone = payload.clone();
        assert_eq!(payload.identity(), clone.identity());
        assert_eq!(budget.snapshot().live_bytes, 12);
        assert_eq!(budget.snapshot().allocations, 1);
        assert!(PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Nv12 {
                y: PlanarPlaneInput::new(&y, 6),
                uv: PlanarPlaneInput::new(&uv, 6),
            },
            &budget,
        )
        .is_err());
        drop(payload);
        assert_eq!(budget.snapshot().live_bytes, 12);
        drop(clone);
        assert_eq!(budget.snapshot().live_bytes, 0);
    }

    #[test]
    fn cpu_references_cover_601_709_full_limited_chroma_edges_and_p010_2020_ramp() {
        let budget = PlanarAllocationBudget::new(4096).unwrap();
        let neutral = yuv420_payload(2, 2, &[16, 235, 81, 145], &[128], &[128], &budget);
        let bt601 = neutral
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Smpte170M,
                TransferCharacteristic::Smpte170M,
                ChromaLocation::Left,
            ))
            .unwrap();
        assert_eq!(&bt601.rgba[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);

        // A saturated bar distinguishes the 601 and 709 coefficients;
        // neutral ramps alone do not exercise the matrix law.
        let red_bar = yuv420_payload(2, 2, &[81; 4], &[90], &[240], &budget);
        let red_601 = red_bar
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Smpte170M,
                TransferCharacteristic::Smpte170M,
                ChromaLocation::Left,
            ))
            .unwrap();
        let red_709 = red_bar
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Left,
            ))
            .unwrap();
        assert_eq!(&red_601.rgba[..4], &[254, 0, 0, 255]);
        assert_eq!(&red_709.rgba[..4], &[255, 24, 0, 255]);

        let full = yuv420_payload(2, 2, &[0, 255, 128, 64], &[128], &[128], &budget);
        let bt709 = full
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Center,
            ))
            .unwrap();
        assert_eq!(&bt709.rgba[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);
        assert_eq!(&bt709.rgba[8..12], &[128, 128, 128, 255]);

        let chroma = yuv420_payload(4, 2, &[128; 8], &[16, 240], &[240, 16], &budget);
        let left = chroma
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Left,
            ))
            .unwrap();
        let center = chroma
            .to_rgba8_cpu_reference(color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Center,
            ))
            .unwrap();
        assert_ne!(
            left.rgba, center.rgba,
            "chroma siting must affect edge reconstruction"
        );
        assert_ne!(&left.rgba[..4], &left.rgba[12..16]);

        let p010_layout = PlanarImageLayout::new(PlanarPixelFormat::P010Le, 4, 2).unwrap();
        let y = p010_words(&[64, 256, 512, 940, 64, 256, 512, 940]);
        let uv = p010_words(&[512, 512, 512, 512]);
        let p010 = PlanarImagePayload::from_planes(
            p010_layout,
            PlanarPlaneInputs::P010Le {
                y: PlanarPlaneInput::new(&y, 8),
                uv: PlanarPlaneInput::new(&uv, 8),
            },
            &budget,
        )
        .unwrap();
        let bt2020 = p010
            .to_rgba8_cpu_reference(color(
                10,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt2020Ncl,
                TransferCharacteristic::Bt2020_10,
                ChromaLocation::TopLeft,
            ))
            .unwrap();
        let ramp: Vec<u8> = bt2020
            .rgba
            .chunks_exact(4)
            .take(4)
            .map(|px| px[0])
            .collect();
        assert_eq!(ramp, vec![0, 56, 130, 255]);
        assert!(bt2020
            .rgba
            .chunks_exact(4)
            .all(|px| px[0] == px[1] && px[1] == px[2] && px[3] == 255));

        let colored_y = p010_words(&[512; 4]);
        let colored_uv = p010_words(&[600, 700]);
        let colored_2020 = PlanarImagePayload::from_planes(
            PlanarImageLayout::new(PlanarPixelFormat::P010Le, 2, 2).unwrap(),
            PlanarPlaneInputs::P010Le {
                y: PlanarPlaneInput::new(&colored_y, 4),
                uv: PlanarPlaneInput::new(&colored_uv, 4),
            },
            &budget,
        )
        .unwrap()
        .to_rgba8_cpu_reference(color(
            10,
            SourceColorRange::Limited,
            MatrixCoefficients::Bt2020Ncl,
            TransferCharacteristic::Bt2020_10,
            ChromaLocation::TopLeft,
        ))
        .unwrap();
        assert_eq!(&colored_2020.rgba[..4], &[209, 96, 178, 255]);
    }

    #[test]
    fn legacy_wrapper_is_byte_identity_motion_identity_and_default_policy_exact() {
        let bytes = vec![1, 2, 3, 4, 9, 8, 7, 6];
        let payload = DecodedImagePayload::from_owned_rgba(bytes.clone());
        let payload_id = payload.identity();
        let motion = motion_fixture(41, 9);
        let metadata = FrameMetadata::sanitized(41, Some(900), 3.0, 8.0).with_codec_identity(Some(
            CodecFrameIdentity {
                source_generation: 41,
                pts: 900,
                presentation_ordinal: 9,
            },
        ));
        let wrapped = DecodedDeliveryFrame::from_legacy(DecodedVideoFrame {
            rgba: payload,
            metadata,
            codec_motion: Some(motion.clone()),
        });
        assert_eq!(wrapped.image.identity(), payload_id);
        assert_eq!(wrapped.image.legacy_rgba_bytes().unwrap(), bytes);
        assert_eq!(wrapped.metadata, metadata);
        assert_eq!(wrapped.codec_motion, Some(motion.clone()));

        let round_trip = wrapped.into_legacy().unwrap();
        assert_eq!(round_trip.rgba.identity(), payload_id);
        assert_eq!(round_trip.rgba.as_slice(), bytes);
        assert_eq!(round_trip.metadata, metadata);
        assert_eq!(round_trip.codec_motion, Some(motion));

        assert_eq!(
            PlanarDeliveryPolicy::default(),
            PlanarDeliveryPolicy::LegacyRgba
        );
        let settings: PlanarDeliverySettings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings.policy, PlanarDeliveryPolicy::LegacyRgba);
    }

    #[test]
    fn planar_frame_preserves_generation_pts_and_codec_motion_without_legacy_laundering() {
        let budget = PlanarAllocationBudget::new(64).unwrap();
        let image = yuv420_payload(2, 2, &[16, 32, 64, 128], &[128], &[128], &budget);
        let image_id = image.identity();
        let motion = motion_fixture(77, 4);
        let metadata = FrameMetadata::sanitized(77, Some(44), 0.5, 10.0).with_codec_identity(Some(
            CodecFrameIdentity {
                source_generation: 77,
                pts: 44,
                presentation_ordinal: 4,
            },
        ));
        let frame = DecodedDeliveryFrame {
            image: DecodedImageDelivery::Planar(image),
            metadata,
            codec_motion: Some(motion.clone()),
        };
        assert_eq!(frame.image.identity(), image_id);
        assert_eq!(frame.metadata.source_generation, 77);
        assert_eq!(frame.metadata.codec_identity.unwrap().source_generation, 77);
        assert_eq!(frame.codec_motion.as_ref().unwrap().source_generation, 77);
        let rejected = frame.into_legacy().unwrap_err();
        assert_eq!(rejected.image.identity(), image_id);
        assert_eq!(rejected.metadata, metadata);
        assert_eq!(rejected.codec_motion, Some(motion));
    }

    #[test]
    fn admission_is_opt_in_and_stops_hdr_interlace_and_incomplete_truth() {
        let sdr = color(
            10,
            SourceColorRange::Limited,
            MatrixCoefficients::Bt2020Ncl,
            TransferCharacteristic::Bt2020_10,
            ChromaLocation::Left,
        );
        assert_eq!(
            prototype_delivery_decision(
                PlanarDeliveryPolicy::LegacyRgba,
                PlanarPixelFormat::P010Le,
                sdr,
                SourceFieldOrder::Progressive,
            ),
            PlanarDeliveryDecision::PackedRgbaFallback(PlanarFallbackReason::LegacyPolicy)
        );
        assert_eq!(
            prototype_delivery_decision(
                PlanarDeliveryPolicy::MetadataManaged,
                PlanarPixelFormat::P010Le,
                sdr,
                SourceFieldOrder::Progressive,
            ),
            PlanarDeliveryDecision::PrototypePlanar(PlanarPixelFormat::P010Le)
        );
        let mut hdr = sdr;
        hdr.transfer.value = TransferCharacteristic::Pq;
        assert_eq!(
            prototype_delivery_decision(
                PlanarDeliveryPolicy::MetadataManaged,
                PlanarPixelFormat::P010Le,
                hdr,
                SourceFieldOrder::Progressive,
            ),
            PlanarDeliveryDecision::PackedRgbaFallback(PlanarFallbackReason::HdrToneMapRequired)
        );
        assert_eq!(
            prototype_delivery_decision(
                PlanarDeliveryPolicy::MetadataManaged,
                PlanarPixelFormat::P010Le,
                sdr,
                SourceFieldOrder::TopCodedTopDisplayed,
            ),
            PlanarDeliveryDecision::PackedRgbaFallback(PlanarFallbackReason::Interlaced)
        );
        assert!(matches!(
            PlanarImageLayout::new(PlanarPixelFormat::P010Le, 2, 2).unwrap(),
            PlanarImageLayout { .. }
        ));
        let budget = PlanarAllocationBudget::new(32).unwrap();
        let y = p010_words(&[64; 4]);
        let uv = p010_words(&[512, 512]);
        let payload = PlanarImagePayload::from_planes(
            PlanarImageLayout::new(PlanarPixelFormat::P010Le, 2, 2).unwrap(),
            PlanarPlaneInputs::P010Le {
                y: PlanarPlaneInput::new(&y, 4),
                uv: PlanarPlaneInput::new(&uv, 4),
            },
            &budget,
        )
        .unwrap();
        assert_eq!(
            payload.to_rgba8_cpu_reference(hdr).unwrap_err(),
            PlanarConversionError::HdrToneMapRequired(TransferCharacteristic::Pq)
        );
    }
}
