//! Fixed, bounded GPU -> CPU capture staging shared by live Program recording
//! and stable post-effects scope resampling.
//!
//! Preparation creates every GPU object. A warmed capture only mutates two
//! fixed slots, encodes a texture copy, requests an asynchronous map, and
//! copies completed rows into caller-owned storage. It never waits for the GPU
//! and never allocates a pixel buffer.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::media_safety::SAFE_MEDIA_MAX_RGBA_BYTES;
use crate::program_recorder::{CaptureTarget, RecorderFrameMetadata};

pub(crate) const RECORDER_GPU_READBACK_SLOTS: usize = 2;
pub(crate) const RECORDER_GPU_READBACK_MAX_BYTES: u64 = 128 * 1024 * 1024;
const RGBA_BYTES_PER_PIXEL: u64 = 4;
const COPY_ROW_ALIGNMENT: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;

const SLOT_IDLE: u8 = 0;
const SLOT_RESERVED: u8 = 1;
const SLOT_COPY_PENDING: u8 = 2;
const SLOT_MAP_REQUESTED: u8 = 3;
const SLOT_MAPPED: u8 = 4;
const SLOT_MAP_FAILED: u8 = 5;

/// The capture-time metadata is deliberately opaque to the renderer. Keeping
/// it in the fixed GPU slot prevents an asynchronous completion from being
/// paired with metadata sampled from a later visual generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecorderReadbackTag {
    capture_generation: u64,
    metadata: RecorderFrameMetadata,
}

#[allow(
    dead_code,
    reason = "the stable capture-request envelope retains accessors for Program, Layer, and Group adapters"
)]
impl RecorderReadbackTag {
    /// `capture_generation` is a caller-owned recorder/still/resample session
    /// identity. Zero is valid.
    pub const fn new(capture_generation: u64, metadata: RecorderFrameMetadata) -> Self {
        Self {
            capture_generation,
            metadata,
        }
    }

    pub const fn capture_generation(self) -> u64 {
        self.capture_generation
    }

    pub const fn metadata(self) -> RecorderFrameMetadata {
        self.metadata
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecorderReadbackRequest {
    pub target: CaptureTarget,
    pub tag: RecorderReadbackTag,
}

impl RecorderReadbackRequest {
    pub const fn new(target: CaptureTarget, tag: RecorderReadbackTag) -> Self {
        Self { target, tag }
    }
}

/// Opaque ownership proof for one prepared staging slot. Callers retain it
/// across command submission and then pass it to `map`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecorderReadbackReservation {
    slot: u8,
    sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecorderReadbackAdmission {
    Scheduled(RecorderReadbackReservation),
    Busy,
    Unprepared,
    SourceUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecorderReadbackDropReason {
    MapFailed,
    StaleGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecorderReadbackCaptureStatus {
    Captured,
    SourceUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecorderReadbackCompleted {
    pub request: RecorderReadbackRequest,
    pub dimensions: [u32; 2],
    pub byte_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecorderReadbackPoll {
    Idle,
    Pending,
    Ready(RecorderReadbackCompleted),
    Dropped {
        request: RecorderReadbackRequest,
        reason: RecorderReadbackDropReason,
    },
}

/// Allocation-free observation of the globally oldest slot. `Ready` does not
/// consume or unmap it; the caller may acquire a CPU frame lease and then call
/// `poll_into` exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecorderReadbackReadiness {
    Idle,
    Pending,
    Ready(RecorderReadbackRequest),
    Dropped {
        request: RecorderReadbackRequest,
        reason: RecorderReadbackDropReason,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RecorderReadbackAllocationSnapshot {
    pub buffers: u8,
    pub buffer_bytes: u64,
    pub conversion_textures: u8,
    pub conversion_texture_bytes: u64,
}

impl RecorderReadbackAllocationSnapshot {
    pub const fn total_objects(self) -> u64 {
        self.buffers as u64 + self.conversion_textures as u64
    }

    pub const fn total_bytes(self) -> u64 {
        self.buffer_bytes + self.conversion_texture_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecorderReadbackError {
    ZeroDimensions([u32; 2]),
    DimensionsExceedDevice {
        requested: [u32; 2],
        limit: u32,
    },
    SizeOverflow([u32; 2]),
    FrameTooLarge {
        requested: u64,
        limit: u64,
    },
    BufferTooLarge {
        requested: u64,
        limit: u64,
    },
    AggregateTooLarge {
        requested: u64,
        limit: u64,
    },
    #[allow(
        dead_code,
        reason = "explicit preparation failures remain part of the recorder diagnostic contract"
    )]
    DeviceUnavailable(String),
    UnsupportedTarget(CaptureTarget),
    ResourceCreation(String),
    InvalidReservation,
    CopyNotEncoded,
    DestinationLength {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for RecorderReadbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions(dimensions) => write!(
                formatter,
                "recorder readback dimensions must be nonzero, got {}x{}",
                dimensions[0], dimensions[1]
            ),
            Self::DimensionsExceedDevice { requested, limit } => write!(
                formatter,
                "recorder readback dimensions {}x{} exceed device limit {limit}",
                requested[0], requested[1]
            ),
            Self::SizeOverflow(dimensions) => write!(
                formatter,
                "recorder readback byte accounting overflowed at {}x{}",
                dimensions[0], dimensions[1]
            ),
            Self::FrameTooLarge { requested, limit } => write!(
                formatter,
                "recorder readback frame needs {requested} bytes; limit is {limit}"
            ),
            Self::BufferTooLarge { requested, limit } => write!(
                formatter,
                "recorder readback staging buffer needs {requested} bytes; device limit is {limit}"
            ),
            Self::AggregateTooLarge { requested, limit } => write!(
                formatter,
                "recorder readback prepared resources need {requested} bytes; limit is {limit}"
            ),
            Self::DeviceUnavailable(message) => {
                write!(
                    formatter,
                    "recorder readback device is unavailable: {message}"
                )
            }
            Self::UnsupportedTarget(target) => {
                write!(
                    formatter,
                    "recorder readback target {target:?} is unsupported here"
                )
            }
            Self::ResourceCreation(message) => {
                write!(
                    formatter,
                    "recorder readback GPU preparation failed: {message}"
                )
            }
            Self::InvalidReservation => {
                formatter.write_str("recorder readback reservation is stale or invalid")
            }
            Self::CopyNotEncoded => formatter.write_str(
                "recorder readback map was requested before its texture copy was encoded",
            ),
            Self::DestinationLength { expected, actual } => write!(
                formatter,
                "recorder readback destination has {actual} bytes; expected exactly {expected}"
            ),
        }
    }
}

impl std::error::Error for RecorderReadbackError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecorderReadbackLayout {
    pub dimensions: [u32; 2],
    pub row_bytes: u32,
    pub padded_row_bytes: u32,
    pub frame_bytes: usize,
    pub buffer_bytes: u64,
}

impl RecorderReadbackLayout {
    pub(crate) fn checked(
        dimensions: [u32; 2],
        device_limits: &wgpu::Limits,
        conversion_texture_count: u8,
    ) -> Result<Self, RecorderReadbackError> {
        if dimensions.contains(&0) {
            return Err(RecorderReadbackError::ZeroDimensions(dimensions));
        }
        if dimensions[0] > device_limits.max_texture_dimension_2d
            || dimensions[1] > device_limits.max_texture_dimension_2d
        {
            return Err(RecorderReadbackError::DimensionsExceedDevice {
                requested: dimensions,
                limit: device_limits.max_texture_dimension_2d,
            });
        }
        let row_bytes = u64::from(dimensions[0])
            .checked_mul(RGBA_BYTES_PER_PIXEL)
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        let frame_bytes = row_bytes
            .checked_mul(u64::from(dimensions[1]))
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        if frame_bytes > SAFE_MEDIA_MAX_RGBA_BYTES {
            return Err(RecorderReadbackError::FrameTooLarge {
                requested: frame_bytes,
                limit: SAFE_MEDIA_MAX_RGBA_BYTES,
            });
        }
        let padded_row_bytes = row_bytes
            .checked_add(COPY_ROW_ALIGNMENT - 1)
            .map(|value| value & !(COPY_ROW_ALIGNMENT - 1))
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        let buffer_bytes = padded_row_bytes
            .checked_mul(u64::from(dimensions[1]))
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        if buffer_bytes > device_limits.max_buffer_size {
            return Err(RecorderReadbackError::BufferTooLarge {
                requested: buffer_bytes,
                limit: device_limits.max_buffer_size,
            });
        }
        let staging_bytes = buffer_bytes
            .checked_mul(RECORDER_GPU_READBACK_SLOTS as u64)
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        let conversion_bytes = frame_bytes
            .checked_mul(u64::from(conversion_texture_count))
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        let aggregate = staging_bytes
            .checked_add(conversion_bytes)
            .ok_or(RecorderReadbackError::SizeOverflow(dimensions))?;
        if aggregate > RECORDER_GPU_READBACK_MAX_BYTES {
            return Err(RecorderReadbackError::AggregateTooLarge {
                requested: aggregate,
                limit: RECORDER_GPU_READBACK_MAX_BYTES,
            });
        }
        Ok(Self {
            dimensions,
            row_bytes: u32::try_from(row_bytes)
                .map_err(|_| RecorderReadbackError::SizeOverflow(dimensions))?,
            padded_row_bytes: u32::try_from(padded_row_bytes)
                .map_err(|_| RecorderReadbackError::SizeOverflow(dimensions))?,
            frame_bytes: usize::try_from(frame_bytes)
                .map_err(|_| RecorderReadbackError::SizeOverflow(dimensions))?,
            buffer_bytes,
        })
    }
}

struct RecorderReadbackSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    sequence: u64,
    request: Option<RecorderReadbackRequest>,
}

pub(crate) struct PreparedRgbaReadback {
    layout: RecorderReadbackLayout,
    slots: [RecorderReadbackSlot; RECORDER_GPU_READBACK_SLOTS],
    next_sequence: u64,
}

impl PreparedRgbaReadback {
    pub(crate) fn prepare(
        device: &wgpu::Device,
        dimensions: [u32; 2],
        conversion_texture_count: u8,
    ) -> Result<Self, RecorderReadbackError> {
        let layout = RecorderReadbackLayout::checked(
            dimensions,
            &device.limits(),
            conversion_texture_count,
        )?;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let slots = std::array::from_fn(|_| RecorderReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Prepared recorder RGBA readback"),
                size: layout.buffer_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            status: Arc::new(AtomicU8::new(SLOT_IDLE)),
            sequence: 0,
            request: None,
        });
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
            layout,
            slots,
            next_sequence: 1,
        })
    }

    pub(crate) const fn allocation_snapshot(
        &self,
        conversion_texture_count: u8,
    ) -> RecorderReadbackAllocationSnapshot {
        RecorderReadbackAllocationSnapshot {
            buffers: RECORDER_GPU_READBACK_SLOTS as u8,
            buffer_bytes: self.layout.buffer_bytes * RECORDER_GPU_READBACK_SLOTS as u64,
            conversion_textures: conversion_texture_count,
            conversion_texture_bytes: self.layout.frame_bytes as u64
                * conversion_texture_count as u64,
        }
    }

    pub(crate) fn reserve(
        &mut self,
        request: RecorderReadbackRequest,
    ) -> RecorderReadbackAdmission {
        let Some((slot_index, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.status.load(Ordering::Acquire) == SLOT_IDLE)
        else {
            return RecorderReadbackAdmission::Busy;
        };
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        slot.sequence = sequence;
        slot.request = Some(request);
        slot.status.store(SLOT_RESERVED, Ordering::Release);
        RecorderReadbackAdmission::Scheduled(RecorderReadbackReservation {
            slot: slot_index as u8,
            sequence,
        })
    }

    pub(crate) fn encode_reserved(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        reservation: RecorderReadbackReservation,
        source: &wgpu::Texture,
    ) -> Result<(), RecorderReadbackError> {
        let layout = self.layout;
        let slot = self.slot_for_mut(reservation)?;
        if slot.status.load(Ordering::Acquire) != SLOT_RESERVED {
            return Err(RecorderReadbackError::InvalidReservation);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_row_bytes),
                    rows_per_image: Some(layout.dimensions[1]),
                },
            },
            wgpu::Extent3d {
                width: layout.dimensions[0],
                height: layout.dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        slot.status.store(SLOT_COPY_PENDING, Ordering::Release);
        Ok(())
    }

    pub(crate) fn schedule_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        request: RecorderReadbackRequest,
    ) -> Result<RecorderReadbackAdmission, RecorderReadbackError> {
        let admission = self.reserve(request);
        if let RecorderReadbackAdmission::Scheduled(reservation) = admission {
            self.encode_reserved(encoder, reservation, source)?;
        }
        Ok(admission)
    }

    /// Request the asynchronous map only after the command buffer containing
    /// the matching copy has been submitted. This method never polls or waits.
    pub(crate) fn map(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        let slot = self.slot_for(reservation)?;
        if slot.status.load(Ordering::Acquire) != SLOT_COPY_PENDING {
            return Err(RecorderReadbackError::CopyNotEncoded);
        }
        slot.status.store(SLOT_MAP_REQUESTED, Ordering::Release);
        let status = Arc::clone(&slot.status);
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(
                    if result.is_ok() {
                        SLOT_MAPPED
                    } else {
                        SLOT_MAP_FAILED
                    },
                    Ordering::Release,
                );
            });
        Ok(())
    }

    /// Recycle a reservation only when the command encoder containing its copy
    /// will not be submitted. Reusing an encoded buffer after submission but
    /// before map completion would violate wgpu ownership.
    pub(crate) fn discard_unsubmitted(
        &mut self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        let slot = self.slot_for_mut(reservation)?;
        if !matches!(
            slot.status.load(Ordering::Acquire),
            SLOT_RESERVED | SLOT_COPY_PENDING
        ) {
            return Err(RecorderReadbackError::InvalidReservation);
        }
        slot.request = None;
        slot.status.store(SLOT_IDLE, Ordering::Release);
        Ok(())
    }

    /// Observe the oldest slot without copying pixels, consuming metadata, or
    /// changing its map state.
    pub(crate) fn oldest_readiness(&self) -> RecorderReadbackReadiness {
        let Some(slot) = self
            .slots
            .iter()
            .filter(|slot| slot.status.load(Ordering::Acquire) != SLOT_IDLE)
            .min_by_key(|slot| slot.sequence)
        else {
            return RecorderReadbackReadiness::Idle;
        };
        match slot.status.load(Ordering::Acquire) {
            SLOT_RESERVED | SLOT_COPY_PENDING | SLOT_MAP_REQUESTED => {
                RecorderReadbackReadiness::Pending
            }
            SLOT_MAPPED => RecorderReadbackReadiness::Ready(
                slot.request
                    .expect("non-idle recorder readback slot has metadata"),
            ),
            SLOT_MAP_FAILED => RecorderReadbackReadiness::Dropped {
                request: slot
                    .request
                    .expect("non-idle recorder readback slot has metadata"),
                reason: RecorderReadbackDropReason::MapFailed,
            },
            _ => unreachable!("unknown recorder readback slot state"),
        }
    }

    /// Recycle the oldest mapped/failed completion without copying its full
    /// frame. Pending GPU work is never cancelled or reused. This is reserved
    /// for a caller-verified stale capture generation.
    pub(crate) fn recycle_oldest_ready_without_copy(
        &mut self,
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        let Some(oldest_index) = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.status.load(Ordering::Acquire) != SLOT_IDLE)
            .min_by_key(|(_, slot)| slot.sequence)
            .map(|(index, _)| index)
        else {
            return Ok(RecorderReadbackPoll::Idle);
        };
        let slot = &mut self.slots[oldest_index];
        match slot.status.load(Ordering::Acquire) {
            SLOT_RESERVED | SLOT_COPY_PENDING | SLOT_MAP_REQUESTED => {
                Ok(RecorderReadbackPoll::Pending)
            }
            SLOT_MAPPED => {
                slot.buffer.unmap();
                let request = slot
                    .request
                    .take()
                    .expect("non-idle recorder readback slot has metadata");
                slot.status.store(SLOT_IDLE, Ordering::Release);
                Ok(RecorderReadbackPoll::Dropped {
                    request,
                    reason: RecorderReadbackDropReason::StaleGeneration,
                })
            }
            SLOT_MAP_FAILED => {
                let request = slot
                    .request
                    .take()
                    .expect("non-idle recorder readback slot has metadata");
                slot.status.store(SLOT_IDLE, Ordering::Release);
                Ok(RecorderReadbackPoll::Dropped {
                    request,
                    reason: RecorderReadbackDropReason::MapFailed,
                })
            }
            _ => unreachable!("unknown recorder readback slot state"),
        }
    }

    /// Harvest only the globally oldest in-flight capture. Callback reordering
    /// cannot reorder CFR input. The destination remains caller-owned.
    pub(crate) fn poll_into(
        &mut self,
        destination: &mut [u8],
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        let Some(oldest_index) = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.status.load(Ordering::Acquire) != SLOT_IDLE)
            .min_by_key(|(_, slot)| slot.sequence)
            .map(|(index, _)| index)
        else {
            return Ok(RecorderReadbackPoll::Idle);
        };
        let slot = &mut self.slots[oldest_index];
        match slot.status.load(Ordering::Acquire) {
            SLOT_RESERVED | SLOT_COPY_PENDING | SLOT_MAP_REQUESTED => {
                Ok(RecorderReadbackPoll::Pending)
            }
            SLOT_MAPPED => {
                if destination.len() != self.layout.frame_bytes {
                    return Err(RecorderReadbackError::DestinationLength {
                        expected: self.layout.frame_bytes,
                        actual: destination.len(),
                    });
                }
                let data = slot.buffer.slice(..).get_mapped_range();
                let row_bytes = self.layout.row_bytes as usize;
                let padded_row_bytes = self.layout.padded_row_bytes as usize;
                for row in 0..self.layout.dimensions[1] as usize {
                    let source_start = row * padded_row_bytes;
                    let target_start = row * row_bytes;
                    destination[target_start..target_start + row_bytes]
                        .copy_from_slice(&data[source_start..source_start + row_bytes]);
                }
                drop(data);
                slot.buffer.unmap();
                let request = slot
                    .request
                    .take()
                    .expect("non-idle recorder readback slot has metadata");
                slot.status.store(SLOT_IDLE, Ordering::Release);
                Ok(RecorderReadbackPoll::Ready(RecorderReadbackCompleted {
                    request,
                    dimensions: self.layout.dimensions,
                    byte_len: self.layout.frame_bytes,
                }))
            }
            SLOT_MAP_FAILED => {
                let request = slot
                    .request
                    .take()
                    .expect("non-idle recorder readback slot has metadata");
                slot.status.store(SLOT_IDLE, Ordering::Release);
                Ok(RecorderReadbackPoll::Dropped {
                    request,
                    reason: RecorderReadbackDropReason::MapFailed,
                })
            }
            _ => unreachable!("unknown recorder readback slot state"),
        }
    }

    fn slot_for(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<&RecorderReadbackSlot, RecorderReadbackError> {
        self.slots
            .get(reservation.slot as usize)
            .filter(|slot| {
                slot.sequence == reservation.sequence
                    && slot.status.load(Ordering::Acquire) != SLOT_IDLE
            })
            .ok_or(RecorderReadbackError::InvalidReservation)
    }

    fn slot_for_mut(
        &mut self,
        reservation: RecorderReadbackReservation,
    ) -> Result<&mut RecorderReadbackSlot, RecorderReadbackError> {
        self.slots
            .get_mut(reservation.slot as usize)
            .filter(|slot| {
                slot.sequence == reservation.sequence
                    && slot.status.load(Ordering::Acquire) != SLOT_IDLE
            })
            .ok_or(RecorderReadbackError::InvalidReservation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(capture_index: u64) -> RecorderFrameMetadata {
        RecorderFrameMetadata {
            capture_index,
            capture_time_ns: capture_index * 1_000,
            program_time_ns: capture_index * 2_000,
            visual_epoch: 7,
            program_frozen: false,
            media_frozen: false,
            blackout: false,
            audio_clock: None,
        }
    }

    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_texture_dimension_2d: 8_192,
            max_buffer_size: 256 * 1024 * 1024,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn layout_is_padded_bounded_and_charges_optional_conversion_texture() {
        let layout = RecorderReadbackLayout::checked([65, 3], &limits(), 1).unwrap();
        assert_eq!(layout.row_bytes, 260);
        assert_eq!(layout.padded_row_bytes, 512);
        assert_eq!(layout.frame_bytes, 780);
        assert_eq!(layout.buffer_bytes, 1_536);

        let uhd = RecorderReadbackLayout::checked([3_840, 2_160], &limits(), 1).unwrap();
        let aggregate =
            uhd.buffer_bytes * RECORDER_GPU_READBACK_SLOTS as u64 + uhd.frame_bytes as u64;
        assert!(aggregate <= RECORDER_GPU_READBACK_MAX_BYTES);
        assert_eq!(uhd.frame_bytes as u64, SAFE_MEDIA_MAX_RGBA_BYTES);
    }

    #[test]
    fn hostile_dimensions_fail_before_any_gpu_allocation() {
        assert!(matches!(
            RecorderReadbackLayout::checked([0, 1], &limits(), 0),
            Err(RecorderReadbackError::ZeroDimensions([0, 1]))
        ));
        assert!(matches!(
            RecorderReadbackLayout::checked([3_841, 2_160], &limits(), 0),
            Err(RecorderReadbackError::FrameTooLarge { .. })
        ));
        let mut narrow = limits();
        narrow.max_buffer_size = 1_024;
        assert!(matches!(
            RecorderReadbackLayout::checked([256, 2], &narrow, 0),
            Err(RecorderReadbackError::BufferTooLarge { .. })
        ));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn prepared_pool_is_fifo_nonblocking_and_reuses_caller_storage() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for prepared readback test");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Prepared readback test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for prepared readback test");
        let dimensions = [4, 2];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prepared readback color source"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let held = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prepared readback held audience"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut pool = PreparedRgbaReadback::prepare(&device, dimensions, 0).unwrap();
        let warmed = pool.allocation_snapshot(0);
        let encode = |pool: &mut PreparedRgbaReadback,
                      rgba: [u8; 4],
                      frame_metadata: RecorderFrameMetadata|
         -> RecorderReadbackReservation {
            let pixels = rgba
                .into_iter()
                .cycle()
                .take(dimensions[0] as usize * dimensions[1] as usize * 4)
                .collect::<Vec<_>>();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
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
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Prepared readback test encoder"),
            });
            let RecorderReadbackAdmission::Scheduled(reservation) = pool
                .schedule_texture(
                    &mut encoder,
                    &texture,
                    RecorderReadbackRequest::new(
                        CaptureTarget::Program,
                        RecorderReadbackTag::new(11, frame_metadata),
                    ),
                )
                .unwrap()
            else {
                panic!("prepared readback slot must be free")
            };
            queue.submit(std::iter::once(encoder.finish()));
            pool.map(reservation).unwrap();
            reservation
        };
        // Prime the held audience with red. The subsequent capture is tagged
        // Program-frozen and must preserve the capture-time tag unchanged.
        let red = [255, 0, 0, 255];
        let red_bytes = red
            .into_iter()
            .cycle()
            .take(dimensions[0] as usize * dimensions[1] as usize * 4)
            .collect::<Vec<_>>();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &red_bytes,
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
        let mut hold_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Prepared readback held-audience prime"),
        });
        hold_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &held,
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
        queue.submit(std::iter::once(hold_encoder.finish()));
        let mut frozen = metadata(1);
        frozen.program_frozen = true;
        let _first = encode(&mut pool, red, frozen);
        let mut blackout = metadata(2);
        blackout.blackout = true;
        let _second = encode(&mut pool, [0, 0, 0, 255], blackout);
        assert_eq!(pool.allocation_snapshot(0), warmed);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("prepared readback wait");
        assert!(matches!(
            pool.oldest_readiness(),
            RecorderReadbackReadiness::Ready(request)
                if request.tag.metadata().capture_index == 1
        ));

        let mut destination = [0_u8; 32];
        let RecorderReadbackPoll::Ready(first) = pool.poll_into(&mut destination).unwrap() else {
            panic!("oldest readback must complete first")
        };
        assert_eq!(first.request.tag.metadata().capture_index, 1);
        assert_eq!(first.request.tag.capture_generation(), 11);
        assert!(first.request.tag.metadata().program_frozen);
        assert!(destination
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 0, 0, 255]));

        let RecorderReadbackPoll::Ready(second) = pool.poll_into(&mut destination).unwrap() else {
            panic!("second readback must complete after first")
        };
        assert_eq!(second.request.tag.metadata().capture_index, 2);
        assert!(second.request.tag.metadata().blackout);
        assert!(destination
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 0, 0, 255]));
        assert_eq!(
            pool.poll_into(&mut destination).unwrap(),
            RecorderReadbackPoll::Idle
        );
        assert_eq!(pool.oldest_readiness(), RecorderReadbackReadiness::Idle);

        // Restoring the held texture before the capture copy yields the exact
        // held red pixels. No fallback or re-render participates.
        let mut restore_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Prepared readback held-audience restore"),
        });
        restore_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &held,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
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
        let RecorderReadbackAdmission::Scheduled(restored) = pool
            .schedule_texture(
                &mut restore_encoder,
                &texture,
                RecorderReadbackRequest::new(
                    CaptureTarget::Program,
                    RecorderReadbackTag::new(11, metadata(3)),
                ),
            )
            .unwrap()
        else {
            panic!("restored capture slot must be free")
        };
        queue.submit(std::iter::once(restore_encoder.finish()));
        pool.map(restored).unwrap();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("held restore readback wait");
        let RecorderReadbackPoll::Ready(restored) = pool.poll_into(&mut destination).unwrap()
        else {
            panic!("held restore capture must complete")
        };
        assert_eq!(restored.request.tag.metadata().capture_index, 3);
        assert!(destination
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 0, 0, 255]));

        // Media Freeze does not hold Program: the exact materialized final
        // image may still change, and its tag remains paired with those bytes.
        let mut media_frozen = metadata(4);
        media_frozen.media_frozen = true;
        let fourth = encode(&mut pool, [0, 255, 0, 255], media_frozen);
        let _ = fourth;
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("media-freeze readback wait");
        let RecorderReadbackPoll::Ready(media) = pool.poll_into(&mut destination).unwrap() else {
            panic!("media-freeze capture must complete")
        };
        assert!(media.request.tag.metadata().media_frozen);
        assert!(destination
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 255, 0, 255]));

        let stale = encode(&mut pool, [0, 0, 255, 255], metadata(5));
        let _ = stale;
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("stale-generation readback wait");
        assert!(matches!(
            pool.oldest_readiness(),
            RecorderReadbackReadiness::Ready(request)
                if request.tag.metadata().capture_index == 5
        ));
        assert!(matches!(
            pool.recycle_oldest_ready_without_copy().unwrap(),
            RecorderReadbackPoll::Dropped {
                request,
                reason: RecorderReadbackDropReason::StaleGeneration,
            } if request.tag.metadata().capture_index == 5
        ));
        assert_eq!(pool.oldest_readiness(), RecorderReadbackReadiness::Idle);
        assert_eq!(pool.allocation_snapshot(0), warmed);
    }
}
