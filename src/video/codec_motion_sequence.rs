//! Bounded skipped-frame composition for codec motion products.
//!
//! A sequence contains every decoded transition after the last accepted image
//! and through the newest image. It accepts only contiguous, same-generation,
//! single-reference-step products; any intra/future-only/malformed/gapped
//! transition makes codec motion unavailable so Auto can visibly fall back to
//! Motion Lattice. Composition traces destination coordinates backwards and
//! combines displacement before returning canonical velocity.

use std::fmt;
use std::mem::size_of;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::motion::{
    rasterize_codec_motion_vectors, CodecMotionVector, CodecReferenceDirection, MotionField,
    MotionFieldError, MotionFieldOrigin, MotionGrid, MotionLatticeQuality, MotionVectorSample,
    MOTION_ALGORITHM_VERSION, MOTION_CODEC_VECTOR_MAX_RECORDS,
};

use super::{CodecMotionFrame, CodecMotionFrameType, CodecMotionStatus};

pub const MAX_CODEC_MOTION_COMPOSITION_FRAMES: usize = 16;
/// The owned sparse chain may never exceed the established one-frame record
/// cap. This prevents a burst of skipped frames multiplying decoder memory.
pub const MAX_CODEC_MOTION_COMPOSITION_RECORDS: usize = MOTION_CODEC_VECTOR_MAX_RECORDS;
/// At most accumulated + next + output f32 grids coexist while composing.
/// Sparse codec steps must remain unquantized until the complete displacement
/// is divided by total elapsed time and packed exactly once.
pub const MAX_CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CODEC_BLOCK_PIXELS: u16 = 256;
const MAX_CODEC_REFERENCE_SECONDS: f32 = 10.0;
static CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocation-free identity for one exactly proven codec-motion interval.
/// Downstream caches and sidecars can hash this value without treating a
/// generation/ordinal/count tuple as proof of the represented reference law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodecMotionProductIdentity {
    pub policy: super::AdjacentReferencePolicy,
    pub first_reference: super::CodecFrameIdentity,
    pub latest_destination: super::CodecFrameIdentity,
    pub total_elapsed_ticks: i64,
    pub time_base: super::CodecTimeBase,
    pub transition_count: u8,
    /// Canonical digest of every validated proof and sparse vector record.
    /// Endpoints/count alone cannot distinguish VFR chains with different
    /// intermediate pictures or products with different vector payloads.
    pub content_sha256: [u8; 32],
}

struct CompositionTransientLease {
    bytes: u64,
}

impl CompositionTransientLease {
    fn try_acquire(bytes: u64) -> Result<Self, CodecMotionSequenceError> {
        let mut observed = CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES.load(Ordering::Acquire);
        loop {
            let requested = observed
                .checked_add(bytes)
                .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?;
            if requested > MAX_CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES {
                return Err(CodecMotionSequenceError::TransientBytes);
            }
            match CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES.compare_exchange_weak(
                observed,
                requested,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(Self { bytes }),
                Err(actual) => observed = actual,
            }
        }
    }
}

impl Drop for CompositionTransientLease {
    fn drop(&mut self) {
        CODEC_MOTION_COMPOSITION_TRANSIENT_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Decoder product paired with one accepted RGBA image. A direct product
/// retains the frozen one-frame adapter; a sequence owns every adjacent
/// transition after the previously accepted image so a dropped presentation
/// frame cannot silently shorten motion.
#[derive(Debug, Clone, PartialEq)]
pub enum CodecMotionProduct {
    Direct(CodecMotionFrame),
    Composed(CodecMotionSequence),
}

impl CodecMotionProduct {
    pub fn latest(&self) -> &CodecMotionFrame {
        match self {
            Self::Direct(frame) => frame,
            Self::Composed(sequence) => sequence
                .latest()
                .expect("a constructed codec motion sequence is non-empty"),
        }
    }

    pub fn transition_count(&self) -> usize {
        self.exact_identity()
            .map_or(0, |identity| usize::from(identity.transition_count))
    }

    /// Return a cache-safe identity only when every transition has a valid
    /// adjacent proof and the complete chain uses one reference policy and
    /// one exact stream time base.
    pub fn exact_identity(&self) -> Option<CodecMotionProductIdentity> {
        match self {
            Self::Direct(frame) => exact_product_identity(std::slice::from_ref(frame)).ok(),
            Self::Composed(sequence) => sequence.exact_identity(),
        }
    }

    /// Exact decoded reference interval represented by this product. Direct
    /// products retain the decoder's one-frame delta; composed products sum
    /// every validated adjacent transition before velocity normalization.
    pub fn elapsed_seconds(&self) -> Option<f32> {
        exact_product_elapsed_seconds(self.exact_identity()?)
    }

    pub fn rasterize(
        &self,
        quality: MotionLatticeQuality,
    ) -> Result<Option<MotionField>, CodecMotionSequenceError> {
        match self {
            Self::Direct(frame) => {
                if frame.status != CodecMotionStatus::Available {
                    return Ok(None);
                }
                validate_frame(frame)?;
                rasterize_codec_motion_vectors(frame.source_dimensions, quality, &frame.vectors)
                    .map_err(CodecMotionSequenceError::Motion)
            }
            Self::Composed(sequence) => sequence.rasterize_composed(quality),
        }
    }

    pub(crate) fn retag_source_generation(&mut self, source_generation: u64) {
        match self {
            Self::Direct(frame) => frame.retag_source_generation(source_generation),
            Self::Composed(sequence) => {
                for frame in &mut sequence.frames {
                    frame.retag_source_generation(source_generation);
                }
            }
        }
    }
}

impl From<CodecMotionFrame> for CodecMotionProduct {
    fn from(frame: CodecMotionFrame) -> Self {
        Self::Direct(frame)
    }
}

impl From<CodecMotionSequence> for CodecMotionProduct {
    fn from(sequence: CodecMotionSequence) -> Self {
        Self::Composed(sequence)
    }
}

impl Deref for CodecMotionProduct {
    type Target = CodecMotionFrame;

    fn deref(&self) -> &Self::Target {
        self.latest()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecMotionSequence {
    frames: Vec<CodecMotionFrame>,
    total_records: usize,
}

impl CodecMotionSequence {
    pub fn from_frame(frame: CodecMotionFrame) -> Result<Self, CodecMotionSequenceError> {
        validate_frame(&frame)?;
        let total_records = frame.vectors.len();
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(1)
            .map_err(|_| CodecMotionSequenceError::Allocation)?;
        frames.push(frame);
        Ok(Self {
            frames,
            total_records,
        })
    }

    pub fn push_contiguous(
        &mut self,
        frame: CodecMotionFrame,
    ) -> Result<(), CodecMotionSequenceError> {
        validate_frame(&frame)?;
        if self.frames.len() >= MAX_CODEC_MOTION_COMPOSITION_FRAMES {
            return Err(CodecMotionSequenceError::FrameCap);
        }
        let previous = self.frames.last().ok_or(CodecMotionSequenceError::Empty)?;
        if frame.source_generation != previous.source_generation {
            return Err(CodecMotionSequenceError::GenerationMismatch);
        }
        if frame.source_dimensions != previous.source_dimensions {
            return Err(CodecMotionSequenceError::DimensionMismatch);
        }
        if frame.algorithm_version != previous.algorithm_version {
            return Err(CodecMotionSequenceError::AlgorithmMismatch);
        }
        if frame.frame_ordinal != previous.frame_ordinal.saturating_add(1) {
            return Err(CodecMotionSequenceError::OrdinalGap {
                previous: previous.frame_ordinal,
                next: frame.frame_ordinal,
            });
        }
        let previous_proof = previous.past_reference_proof.ok_or(
            CodecMotionSequenceError::NonAdjacentReference {
                ordinal: previous.frame_ordinal,
            },
        )?;
        let next_proof =
            frame
                .past_reference_proof
                .ok_or(CodecMotionSequenceError::NonAdjacentReference {
                    ordinal: frame.frame_ordinal,
                })?;
        let previous_time_base = canonical_time_base(previous_proof.time_base).ok_or(
            CodecMotionSequenceError::ProofLawMismatch {
                ordinal: previous.frame_ordinal,
            },
        )?;
        let next_time_base = canonical_time_base(next_proof.time_base).ok_or(
            CodecMotionSequenceError::ProofLawMismatch {
                ordinal: frame.frame_ordinal,
            },
        )?;
        if previous_proof.policy != next_proof.policy || previous_time_base != next_time_base {
            return Err(CodecMotionSequenceError::ProofLawMismatch {
                ordinal: frame.frame_ordinal,
            });
        }
        if previous_proof.destination != next_proof.reference {
            return Err(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: frame.frame_ordinal,
            });
        }
        let total_records = self
            .total_records
            .checked_add(frame.vectors.len())
            .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?;
        if total_records > MAX_CODEC_MOTION_COMPOSITION_RECORDS {
            return Err(CodecMotionSequenceError::RecordCap);
        }
        let owned_bytes = total_records
            .checked_mul(size_of::<CodecMotionVector>())
            .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?;
        if owned_bytes > super::codec_motion::MAX_CODEC_MOTION_OWNED_BYTES {
            return Err(CodecMotionSequenceError::OwnedBytes);
        }
        self.frames
            .try_reserve(1)
            .map_err(|_| CodecMotionSequenceError::Allocation)?;
        self.frames.push(frame);
        self.total_records = total_records;
        Ok(())
    }

    pub fn latest(&self) -> Option<&CodecMotionFrame> {
        self.frames.last()
    }

    pub fn exact_identity(&self) -> Option<CodecMotionProductIdentity> {
        exact_product_identity(&self.frames).ok()
    }

    pub fn rasterize_composed(
        &self,
        quality: MotionLatticeQuality,
    ) -> Result<Option<MotionField>, CodecMotionSequenceError> {
        let exact_identity = exact_product_identity(&self.frames)?;
        let first = self.frames.first().ok_or(CodecMotionSequenceError::Empty)?;
        if self.frames.len() == 1 {
            return rasterize_codec_motion_vectors(
                first.source_dimensions,
                quality,
                &first.vectors,
            )
            .map_err(CodecMotionSequenceError::Motion);
        }
        let grid = MotionGrid::for_source(first.source_dimensions, quality)
            .map_err(|_| CodecMotionSequenceError::Motion(MotionFieldError::GridLimit))?;
        let float_sample_bytes = u64::try_from(size_of::<MotionVectorSample>())
            .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
        let best_area_bytes = u64::try_from(size_of::<u32>())
            .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
        let transient_bytes = grid
            .vector_count
            .checked_mul(
                float_sample_bytes
                    .checked_mul(2)
                    .and_then(|bytes| bytes.checked_add(best_area_bytes))
                    .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?,
            )
            .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?;
        let _transient_lease = CompositionTransientLease::try_acquire(transient_bytes)?;
        let mut accumulated: Option<FloatMotionField> = None;
        for frame in &self.frames {
            let step = rasterize_displacement_vectors(frame, grid)?;
            accumulated = Some(match accumulated {
                None => step,
                Some(previous) => compose_displacement_fields(&previous, step)?,
            });
        }
        let total_seconds = exact_product_elapsed_seconds(exact_identity)
            .ok_or(CodecMotionSequenceError::InvalidElapsed)?;
        let accumulated = accumulated.ok_or(CodecMotionSequenceError::Empty)?;
        let grid = accumulated.grid;
        let count = usize::try_from(grid.vector_count)
            .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
        let samples = (0..count).map(|index| {
            let mut sample = accumulated.samples[index];
            sample.velocity_uv_per_second[0] /= total_seconds;
            sample.velocity_uv_per_second[1] /= total_seconds;
            sample
        });
        MotionField::from_samples(
            accumulated.source_dimensions,
            grid,
            MotionFieldOrigin::CodecVectors,
            samples,
        )
        .map(Some)
        .map_err(CodecMotionSequenceError::Motion)
    }
}

fn exact_product_identity(
    frames: &[CodecMotionFrame],
) -> Result<CodecMotionProductIdentity, CodecMotionSequenceError> {
    let first = frames.first().ok_or(CodecMotionSequenceError::Empty)?;
    validate_frame(first)?;
    let first_proof =
        first
            .past_reference_proof
            .ok_or(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: first.frame_ordinal,
            })?;
    let time_base = canonical_time_base(first_proof.time_base).ok_or(
        CodecMotionSequenceError::ProofLawMismatch {
            ordinal: first.frame_ordinal,
        },
    )?;
    let mut latest_destination = first_proof.reference;
    let mut total_elapsed_ticks = 0_i64;
    let mut content = Sha256::new();
    content.update(b"collide-o-scope/codec-motion-product/v1\0");
    content.update(
        u64::try_from(frames.len())
            .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?
            .to_le_bytes(),
    );
    content.update([match first_proof.policy {
        super::AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp => 0,
    }]);
    for frame in frames {
        validate_frame(frame)?;
        let proof =
            frame
                .past_reference_proof
                .ok_or(CodecMotionSequenceError::NonAdjacentReference {
                    ordinal: frame.frame_ordinal,
                })?;
        if proof.policy != first_proof.policy
            || canonical_time_base(proof.time_base) != Some(time_base)
        {
            return Err(CodecMotionSequenceError::ProofLawMismatch {
                ordinal: frame.frame_ordinal,
            });
        }
        if proof.reference != latest_destination {
            return Err(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: frame.frame_ordinal,
            });
        }
        total_elapsed_ticks = total_elapsed_ticks
            .checked_add(proof.elapsed_ticks)
            .ok_or(CodecMotionSequenceError::ArithmeticOverflow)?;
        latest_destination = proof.destination;

        for value in frame.source_dimensions {
            content.update(value.to_le_bytes());
        }
        content.update(frame.frame_delta_seconds.to_bits().to_le_bytes());
        content.update(frame.source_generation.to_le_bytes());
        content.update(frame.frame_ordinal.to_le_bytes());
        content.update(frame.algorithm_version.to_le_bytes());
        content.update([0]); // FfmpegExportMvs provenance, versioned by the domain tag.
        content.update([1]); // Predictive frame type; validation rejects every other type.
        hash_frame_identity(&mut content, proof.reference);
        hash_frame_identity(&mut content, proof.destination);
        content.update(proof.elapsed_ticks.to_le_bytes());
        content.update(time_base.numerator.to_le_bytes());
        content.update(time_base.denominator.to_le_bytes());
        content.update(
            u64::try_from(frame.vectors.len())
                .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        for vector in &frame.vectors {
            for value in vector.destination {
                content.update(value.to_le_bytes());
            }
            for value in vector.block {
                content.update(value.to_le_bytes());
            }
            for value in vector.motion {
                content.update(value.to_le_bytes());
            }
            content.update(vector.motion_scale.to_le_bytes());
            content.update(vector.seconds_from_reference.to_bits().to_le_bytes());
            content.update([match vector.reference {
                CodecReferenceDirection::Past => 0,
                CodecReferenceDirection::Future => 1,
            }]);
            content.update(vector.visibility.to_bits().to_le_bytes());
        }
    }
    if total_elapsed_ticks <= 0 {
        return Err(CodecMotionSequenceError::InvalidElapsed);
    }
    let transition_count =
        u8::try_from(frames.len()).map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
    Ok(CodecMotionProductIdentity {
        policy: first_proof.policy,
        first_reference: first_proof.reference,
        latest_destination,
        total_elapsed_ticks,
        time_base,
        transition_count,
        content_sha256: content.finalize().into(),
    })
}

fn hash_frame_identity(digest: &mut Sha256, identity: super::CodecFrameIdentity) {
    digest.update(identity.source_generation.to_le_bytes());
    digest.update(identity.pts.to_le_bytes());
    digest.update(identity.presentation_ordinal.to_le_bytes());
}

fn canonical_time_base(time_base: super::CodecTimeBase) -> Option<super::CodecTimeBase> {
    super::CodecTimeBase::new(time_base.numerator, time_base.denominator)
}

fn exact_product_elapsed_seconds(identity: CodecMotionProductIdentity) -> Option<f32> {
    let seconds = identity.total_elapsed_ticks as f64 * f64::from(identity.time_base.numerator)
        / f64::from(identity.time_base.denominator);
    (seconds.is_finite() && seconds > 0.0 && seconds <= f64::from(f32::MAX))
        .then_some(seconds as f32)
}

fn validate_frame(frame: &CodecMotionFrame) -> Result<(), CodecMotionSequenceError> {
    if frame.algorithm_version != MOTION_ALGORITHM_VERSION {
        return Err(CodecMotionSequenceError::AlgorithmMismatch);
    }
    if frame.status != CodecMotionStatus::Available {
        return Err(CodecMotionSequenceError::UnavailableStep {
            ordinal: frame.frame_ordinal,
            status: frame.status,
        });
    }
    if frame.frame_type != CodecMotionFrameType::Predictive {
        return Err(CodecMotionSequenceError::UnsupportedFrameType {
            ordinal: frame.frame_ordinal,
            frame_type: frame.frame_type,
        });
    }
    if frame.vectors.is_empty() || frame.vectors.len() > MOTION_CODEC_VECTOR_MAX_RECORDS {
        return Err(CodecMotionSequenceError::RecordCap);
    }
    if !frame.frame_delta_seconds.is_finite() || frame.frame_delta_seconds <= 0.0 {
        return Err(CodecMotionSequenceError::InvalidElapsed);
    }
    let proof =
        frame
            .past_reference_proof
            .ok_or(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: frame.frame_ordinal,
            })?;
    if proof.destination.source_generation != frame.source_generation
        || proof.destination.presentation_ordinal != frame.frame_ordinal
    {
        return Err(CodecMotionSequenceError::NonAdjacentReference {
            ordinal: frame.frame_ordinal,
        });
    }
    let proven_elapsed =
        proof
            .elapsed_seconds()
            .ok_or(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: frame.frame_ordinal,
            })?;
    let tolerance = (proven_elapsed * 0.001).max(1.0e-6);
    let mut observed_past = false;
    for (index, vector) in frame.vectors.iter().enumerate() {
        validate_sparse_vector(frame, index, *vector)?;
        if vector.reference == CodecReferenceDirection::Future {
            continue;
        }
        observed_past = true;
        if !vector.seconds_from_reference.is_finite()
            || (vector.seconds_from_reference - proven_elapsed).abs() > tolerance
        {
            return Err(CodecMotionSequenceError::NonAdjacentReference {
                ordinal: frame.frame_ordinal,
            });
        }
    }
    if !observed_past {
        return Err(CodecMotionSequenceError::UnavailableStep {
            ordinal: frame.frame_ordinal,
            status: frame.status,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct FloatMotionField {
    source_dimensions: [u32; 2],
    grid: MotionGrid,
    samples: Vec<MotionVectorSample>,
}

fn rasterize_displacement_vectors(
    frame: &CodecMotionFrame,
    grid: MotionGrid,
) -> Result<FloatMotionField, CodecMotionSequenceError> {
    let count = usize::try_from(grid.vector_count)
        .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| CodecMotionSequenceError::Allocation)?;
    samples.resize(count, MotionVectorSample::default());
    let mut best_area = Vec::new();
    best_area
        .try_reserve_exact(count)
        .map_err(|_| CodecMotionSequenceError::Allocation)?;
    best_area.resize(count, u32::MAX);
    let [source_width, source_height] = frame.source_dimensions;
    for (index, vector) in frame.vectors.iter().copied().enumerate() {
        validate_sparse_vector(frame, index, vector)?;
        if vector.reference == CodecReferenceDirection::Future {
            continue;
        }
        let half_width = i64::from(vector.block[0]) / 2;
        let half_height = i64::from(vector.block[1]) / 2;
        let left = i64::from(vector.destination[0]) - half_width;
        let top = i64::from(vector.destination[1]) - half_height;
        let right = left + i64::from(vector.block[0]);
        let bottom = top + i64::from(vector.block[1]);
        let clipped_left = left.clamp(0, i64::from(source_width));
        let clipped_top = top.clamp(0, i64::from(source_height));
        let clipped_right = right.clamp(0, i64::from(source_width));
        let clipped_bottom = bottom.clamp(0, i64::from(source_height));
        if clipped_left >= clipped_right || clipped_top >= clipped_bottom {
            continue;
        }
        let block_pixels = i64::from(grid.block_pixels);
        let grid_left = u32::try_from(clipped_left / block_pixels).unwrap();
        let grid_top = u32::try_from(clipped_top / block_pixels).unwrap();
        let grid_right = u32::try_from((clipped_right + block_pixels - 1) / block_pixels)
            .unwrap()
            .min(grid.width);
        let grid_bottom = u32::try_from((clipped_bottom + block_pixels - 1) / block_pixels)
            .unwrap()
            .min(grid.height);
        let area = u32::from(vector.block[0]) * u32::from(vector.block[1]);
        let sample = MotionVectorSample {
            // Keep exact f32 displacement through every skipped transition;
            // packing here would erase sub-quantum one-pixel motion at HD.
            velocity_uv_per_second: [
                -(vector.motion[0] as f32 / f32::from(vector.motion_scale)) / source_width as f32,
                -(vector.motion[1] as f32 / f32::from(vector.motion_scale)) / source_height as f32,
            ],
            confidence: 1.0,
            visibility: vector.visibility.clamp(0.0, 1.0),
        };
        for y in grid_top..grid_bottom {
            for x in grid_left..grid_right {
                let cell = usize::try_from(u64::from(y) * u64::from(grid.width) + u64::from(x))
                    .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
                if area < best_area[cell] {
                    best_area[cell] = area;
                    samples[cell] = sample;
                }
            }
        }
    }
    Ok(FloatMotionField {
        source_dimensions: frame.source_dimensions,
        grid,
        samples,
    })
}

fn validate_sparse_vector(
    frame: &CodecMotionFrame,
    index: usize,
    vector: CodecMotionVector,
) -> Result<(), CodecMotionSequenceError> {
    let invalid = vector.block[0] == 0
        || vector.block[1] == 0
        || vector.block[0] > MAX_CODEC_BLOCK_PIXELS
        || vector.block[1] > MAX_CODEC_BLOCK_PIXELS
        || vector.motion_scale == 0
        || !vector.seconds_from_reference.is_finite()
        || vector.seconds_from_reference <= 0.0
        || vector.seconds_from_reference > MAX_CODEC_REFERENCE_SECONDS
        || !vector.visibility.is_finite();
    let coordinate_limit = i64::from(frame.source_dimensions[0].max(frame.source_dimensions[1]))
        + i64::from(MAX_CODEC_BLOCK_PIXELS);
    let invalid_coordinate = vector
        .destination
        .iter()
        .any(|coordinate| i64::from(*coordinate).abs() > coordinate_limit);
    let invalid_motion =
        vector
            .motion
            .iter()
            .zip(frame.source_dimensions)
            .any(|(component, dimension)| {
                let displacement = *component as f64 / f64::from(vector.motion_scale.max(1));
                !displacement.is_finite()
                    || displacement.abs()
                        > f64::from(dimension.saturating_add(u32::from(MAX_CODEC_BLOCK_PIXELS)))
            });
    if invalid || invalid_coordinate || invalid_motion {
        return Err(CodecMotionSequenceError::InvalidVector {
            ordinal: frame.frame_ordinal,
            index,
        });
    }
    Ok(())
}

fn compose_displacement_fields(
    previous: &FloatMotionField,
    mut next: FloatMotionField,
) -> Result<FloatMotionField, CodecMotionSequenceError> {
    if previous.source_dimensions != next.source_dimensions || previous.grid != next.grid {
        return Err(CodecMotionSequenceError::DimensionMismatch);
    }
    let grid = next.grid;
    let count = usize::try_from(grid.vector_count)
        .map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
    let width =
        usize::try_from(grid.width).map_err(|_| CodecMotionSequenceError::ArithmeticOverflow)?;
    for index in 0..count {
        let x = u32::try_from(index % width).unwrap();
        let y = u32::try_from(index / width).unwrap();
        let step = next.samples[index];
        let destination_uv = [
            (x as f32 + 0.5) / grid.width as f32,
            (y as f32 + 0.5) / grid.height as f32,
        ];
        let prior_uv = [
            destination_uv[0] - step.velocity_uv_per_second[0],
            destination_uv[1] - step.velocity_uv_per_second[1],
        ];
        let prior = sample_field_linear(previous, prior_uv);
        next.samples[index] = MotionVectorSample {
            velocity_uv_per_second: [
                step.velocity_uv_per_second[0] + prior.velocity_uv_per_second[0],
                step.velocity_uv_per_second[1] + prior.velocity_uv_per_second[1],
            ],
            confidence: step.confidence.min(prior.confidence),
            visibility: step.visibility.min(prior.visibility),
        };
    }
    Ok(next)
}

fn sample_field_linear(field: &FloatMotionField, uv: [f32; 2]) -> MotionVectorSample {
    if uv
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return MotionVectorSample::default();
    }
    let grid = field.grid;
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
        let x = usize::try_from(x.clamp(0, maximum[0])).unwrap();
        let y = usize::try_from(y.clamp(0, maximum[1])).unwrap();
        field.samples[y * usize::try_from(grid.width).unwrap() + x]
    };
    let upper = [base[0] + 1, base[1] + 1];
    let s00 = sample(base[0], base[1]);
    let s10 = sample(upper[0], base[1]);
    let s01 = sample(base[0], upper[1]);
    let s11 = sample(upper[0], upper[1]);
    let bilinear = |values: [f32; 4]| {
        let row0 = values[0] + (values[1] - values[0]) * fraction[0];
        let row1 = values[2] + (values[3] - values[2]) * fraction[0];
        row0 + (row1 - row0) * fraction[1]
    };
    MotionVectorSample {
        velocity_uv_per_second: [
            bilinear([
                s00.velocity_uv_per_second[0],
                s10.velocity_uv_per_second[0],
                s01.velocity_uv_per_second[0],
                s11.velocity_uv_per_second[0],
            ]),
            bilinear([
                s00.velocity_uv_per_second[1],
                s10.velocity_uv_per_second[1],
                s01.velocity_uv_per_second[1],
                s11.velocity_uv_per_second[1],
            ]),
        ],
        confidence: bilinear([
            s00.confidence,
            s10.confidence,
            s01.confidence,
            s11.confidence,
        ]),
        visibility: bilinear([
            s00.visibility,
            s10.visibility,
            s01.visibility,
            s11.visibility,
        ]),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CodecMotionSequenceError {
    Empty,
    FrameCap,
    RecordCap,
    OwnedBytes,
    TransientBytes,
    GenerationMismatch,
    DimensionMismatch,
    AlgorithmMismatch,
    UnsupportedFrameType {
        ordinal: u64,
        frame_type: CodecMotionFrameType,
    },
    OrdinalGap {
        previous: u64,
        next: u64,
    },
    NonAdjacentReference {
        ordinal: u64,
    },
    ProofLawMismatch {
        ordinal: u64,
    },
    UnavailableStep {
        ordinal: u64,
        status: CodecMotionStatus,
    },
    InvalidVector {
        ordinal: u64,
        index: usize,
    },
    InvalidElapsed,
    Allocation,
    ArithmeticOverflow,
    Motion(MotionFieldError),
}

impl fmt::Display for CodecMotionSequenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("codec motion sequence is empty"),
            Self::FrameCap => formatter.write_str("codec motion sequence frame cap reached"),
            Self::RecordCap => formatter.write_str("codec motion sequence record cap reached"),
            Self::OwnedBytes => formatter.write_str("codec motion sequence byte cap reached"),
            Self::TransientBytes => {
                formatter.write_str("codec motion composition transient byte cap reached")
            }
            Self::GenerationMismatch => {
                formatter.write_str("codec motion sequence crosses a source generation")
            }
            Self::DimensionMismatch => {
                formatter.write_str("codec motion sequence dimensions do not match")
            }
            Self::AlgorithmMismatch => {
                formatter.write_str("codec motion sequence algorithm does not match")
            }
            Self::UnsupportedFrameType {
                ordinal,
                frame_type,
            } => write!(
                formatter,
                "codec motion frame {ordinal} has unsupported sequence type {frame_type:?}"
            ),
            Self::OrdinalGap { previous, next } => write!(
                formatter,
                "codec motion sequence ordinal gap {previous} -> {next}"
            ),
            Self::NonAdjacentReference { ordinal } => write!(
                formatter,
                "codec motion frame {ordinal} does not reference the adjacent past frame"
            ),
            Self::ProofLawMismatch { ordinal } => write!(
                formatter,
                "codec motion frame {ordinal} changes the proven reference policy or time base"
            ),
            Self::UnavailableStep { ordinal, status } => {
                write!(
                    formatter,
                    "codec motion frame {ordinal} is unavailable: {status:?}"
                )
            }
            Self::InvalidVector { ordinal, index } => write!(
                formatter,
                "codec motion frame {ordinal} contains invalid vector {index}"
            ),
            Self::InvalidElapsed => formatter.write_str("codec motion elapsed time is invalid"),
            Self::Allocation => {
                formatter.write_str("codec motion composition allocation was rejected")
            }
            Self::ArithmeticOverflow => formatter.write_str("codec motion composition overflow"),
            Self::Motion(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CodecMotionSequenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::CodecReferenceDirection;
    use crate::video::{
        AdjacentReferencePolicy, CodecFrameIdentity, CodecMotionFrameType, CodecMotionProvenance,
        CodecMotionRejectReason, CodecPastReferenceProof, CodecTimeBase,
    };

    fn frame(ordinal: u64, pixels: i32) -> CodecMotionFrame {
        CodecMotionFrame {
            source_dimensions: [64, 64],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation: 7,
            frame_ordinal: ordinal,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            provenance: CodecMotionProvenance::FfmpegExportMvs,
            frame_type: CodecMotionFrameType::Predictive,
            status: CodecMotionStatus::Available,
            past_reference_proof: Some(CodecPastReferenceProof {
                policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
                reference: CodecFrameIdentity {
                    source_generation: 7,
                    pts: i64::try_from(ordinal.saturating_sub(1)).unwrap(),
                    presentation_ordinal: ordinal.saturating_sub(1),
                },
                destination: CodecFrameIdentity {
                    source_generation: 7,
                    pts: i64::try_from(ordinal).unwrap(),
                    presentation_ordinal: ordinal,
                },
                elapsed_ticks: 1,
                time_base: CodecTimeBase::new(1, 30).unwrap(),
            }),
            vectors: vec![CodecMotionVector {
                destination: [32, 32],
                block: [64, 64],
                motion: [-pixels, 0],
                motion_scale: 1,
                seconds_from_reference: 1.0 / 30.0,
                reference: CodecReferenceDirection::Past,
                visibility: 1.0,
            }],
        }
    }

    #[test]
    fn two_and_four_skipped_transitions_compose_displacement_before_velocity() {
        for steps in [2_u64, 4] {
            let mut sequence = CodecMotionSequence::from_frame(frame(10, 1)).unwrap();
            for ordinal in 11..10 + steps {
                sequence.push_contiguous(frame(ordinal, 1)).unwrap();
            }
            let field = sequence
                .rasterize_composed(MotionLatticeQuality::Live)
                .unwrap()
                .unwrap();
            let sample = field.sample(4, 4).unwrap();
            let elapsed = steps as f32 / 30.0;
            let displacement = sample.velocity_uv_per_second[0] * 64.0 * elapsed;
            assert!((displacement - steps as f32).abs() < 0.02, "{steps} steps");
            assert!(sample.confidence > 0.99);
            assert!(sample.visibility > 0.99);
        }
    }

    #[test]
    fn hd_one_pixel_steps_stay_f32_until_the_single_final_pack() {
        let frame = |ordinal| CodecMotionFrame {
            source_dimensions: [1_920, 1_080],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation: 7,
            frame_ordinal: ordinal,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            provenance: CodecMotionProvenance::FfmpegExportMvs,
            frame_type: CodecMotionFrameType::Predictive,
            status: CodecMotionStatus::Available,
            past_reference_proof: Some(CodecPastReferenceProof {
                policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
                reference: CodecFrameIdentity {
                    source_generation: 7,
                    pts: i64::try_from(ordinal.saturating_sub(1)).unwrap(),
                    presentation_ordinal: ordinal.saturating_sub(1),
                },
                destination: CodecFrameIdentity {
                    source_generation: 7,
                    pts: i64::try_from(ordinal).unwrap(),
                    presentation_ordinal: ordinal,
                },
                elapsed_ticks: 1,
                time_base: CodecTimeBase::new(1, 30).unwrap(),
            }),
            vectors: vec![CodecMotionVector {
                destination: [512, 512],
                block: [256, 256],
                motion: [-1, 0],
                motion_scale: 1,
                seconds_from_reference: 1.0 / 30.0,
                reference: CodecReferenceDirection::Past,
                visibility: 1.0,
            }],
        };
        let mut sequence = CodecMotionSequence::from_frame(frame(10)).unwrap();
        sequence.push_contiguous(frame(11)).unwrap();
        let field = sequence
            .rasterize_composed(MotionLatticeQuality::High)
            .unwrap()
            .unwrap();
        let sample = field.sample(128, 128).unwrap();
        let displacement = sample.velocity_uv_per_second[0] * 1_920.0 * (2.0 / 30.0);
        assert!(
            (displacement - 2.0).abs() < 0.3,
            "two one-pixel transitions must not quantize to zero: {displacement}"
        );
    }

    #[test]
    fn direct_sequence_retains_the_frozen_single_frame_adapter() {
        let direct = frame(10, 4);
        let expected =
            rasterize_codec_motion_vectors([64, 64], MotionLatticeQuality::Live, &direct.vectors)
                .unwrap();
        let actual = CodecMotionSequence::from_frame(direct)
            .unwrap()
            .rasterize_composed(MotionLatticeQuality::Live)
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn product_identity_is_exact_hashable_and_vfr_tick_preserving() {
        let time_base = CodecTimeBase::new(1, 1_000).unwrap();
        let mut first = frame(10, 1);
        first.frame_delta_seconds = 0.001;
        first.vectors[0].seconds_from_reference = 0.001;
        first.past_reference_proof = Some(CodecPastReferenceProof {
            policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
            reference: CodecFrameIdentity {
                source_generation: 7,
                pts: 100,
                presentation_ordinal: 9,
            },
            destination: CodecFrameIdentity {
                source_generation: 7,
                pts: 101,
                presentation_ordinal: 10,
            },
            elapsed_ticks: 1,
            time_base,
        });
        let mut second = frame(11, 3);
        second.frame_delta_seconds = 0.003;
        second.vectors[0].seconds_from_reference = 0.003;
        second.past_reference_proof = Some(CodecPastReferenceProof {
            policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
            reference: first.past_reference_proof.unwrap().destination,
            destination: CodecFrameIdentity {
                source_generation: 7,
                pts: 104,
                presentation_ordinal: 11,
            },
            elapsed_ticks: 3,
            time_base,
        });

        let mut sequence = CodecMotionSequence::from_frame(first).unwrap();
        sequence.push_contiguous(second).unwrap();
        let product = CodecMotionProduct::from(sequence);
        let identity = product.exact_identity().unwrap();
        assert_eq!(identity.first_reference.pts, 100);
        assert_eq!(identity.latest_destination.pts, 104);
        assert_eq!(identity.total_elapsed_ticks, 4);
        assert_eq!(identity.time_base, time_base);
        assert_eq!(identity.transition_count, 2);
        assert_ne!(identity.content_sha256, [0; 32]);
        assert!((product.elapsed_seconds().unwrap() - 0.004).abs() < 1.0e-7);
        let mut identities = std::collections::HashSet::new();
        identities.insert(identity);
        assert!(identities.contains(&identity));

        let field = product
            .rasterize(MotionLatticeQuality::Live)
            .unwrap()
            .unwrap();
        let sample = field.sample(4, 4).unwrap();
        let displacement = sample.velocity_uv_per_second[0] * 64.0 * 0.004;
        assert!((displacement - 4.0).abs() < 0.03, "{displacement}");
    }

    #[test]
    fn product_identity_distinguishes_intermediate_proofs_and_vector_payloads() {
        let time_base = CodecTimeBase::new(1, 1_000).unwrap();
        let product = |middle_pts: i64, first_motion: i32| {
            let mut first = frame(10, first_motion);
            let first_ticks = middle_pts - 100;
            first.frame_delta_seconds = first_ticks as f32 / 1_000.0;
            first.vectors[0].seconds_from_reference = first.frame_delta_seconds;
            first.past_reference_proof = Some(CodecPastReferenceProof {
                policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
                reference: CodecFrameIdentity {
                    source_generation: 7,
                    pts: 100,
                    presentation_ordinal: 9,
                },
                destination: CodecFrameIdentity {
                    source_generation: 7,
                    pts: middle_pts,
                    presentation_ordinal: 10,
                },
                elapsed_ticks: first_ticks,
                time_base,
            });
            let mut second = frame(11, 1);
            let second_ticks = 104 - middle_pts;
            second.frame_delta_seconds = second_ticks as f32 / 1_000.0;
            second.vectors[0].seconds_from_reference = second.frame_delta_seconds;
            second.past_reference_proof = Some(CodecPastReferenceProof {
                policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
                reference: first.past_reference_proof.unwrap().destination,
                destination: CodecFrameIdentity {
                    source_generation: 7,
                    pts: 104,
                    presentation_ordinal: 11,
                },
                elapsed_ticks: second_ticks,
                time_base,
            });
            let mut sequence = CodecMotionSequence::from_frame(first).unwrap();
            sequence.push_contiguous(second).unwrap();
            CodecMotionProduct::from(sequence).exact_identity().unwrap()
        };

        let one_then_three = product(101, 1);
        let three_then_one = product(103, 1);
        let changed_vectors = product(101, 2);
        for candidate in [three_then_one, changed_vectors] {
            assert_eq!(candidate.first_reference, one_then_three.first_reference);
            assert_eq!(
                candidate.latest_destination,
                one_then_three.latest_destination
            );
            assert_eq!(
                candidate.total_elapsed_ticks,
                one_then_three.total_elapsed_ticks
            );
            assert_eq!(candidate.time_base, one_then_three.time_base);
            assert_eq!(candidate.transition_count, one_then_three.transition_count);
            assert_ne!(candidate.content_sha256, one_then_three.content_sha256);
            assert_ne!(candidate, one_then_three);
        }
    }

    #[test]
    fn direct_products_cannot_bypass_typed_proof_validation() {
        let mut forged = frame(10, 1);
        forged.past_reference_proof = None;
        let product = CodecMotionProduct::Direct(forged);
        assert_eq!(product.transition_count(), 0);
        assert_eq!(product.exact_identity(), None);
        assert_eq!(
            product.rasterize(MotionLatticeQuality::Live),
            Err(CodecMotionSequenceError::NonAdjacentReference { ordinal: 10 })
        );

        let mut unavailable = frame(10, 1);
        unavailable.status = CodecMotionStatus::ReferenceUnproven;
        unavailable.past_reference_proof = None;
        unavailable.vectors.clear();
        assert_eq!(
            CodecMotionProduct::Direct(unavailable)
                .rasterize(MotionLatticeQuality::Live)
                .unwrap(),
            None
        );
    }

    #[test]
    fn mixed_proof_laws_reject_before_mutating_the_chain() {
        let mut sequence = CodecMotionSequence::from_frame(frame(10, 1)).unwrap();
        let before = sequence.clone();
        let mut changed_time_base = frame(11, 1);
        changed_time_base.frame_delta_seconds = 1.0 / 60.0;
        changed_time_base.vectors[0].seconds_from_reference = 1.0 / 60.0;
        changed_time_base
            .past_reference_proof
            .as_mut()
            .unwrap()
            .time_base = CodecTimeBase::new(1, 60).unwrap();
        assert_eq!(
            sequence.push_contiguous(changed_time_base),
            Err(CodecMotionSequenceError::ProofLawMismatch { ordinal: 11 })
        );
        assert_eq!(sequence, before);
    }

    #[test]
    fn hostile_grid_over_transient_cap_rejects_before_float_allocation() {
        let oversized = |ordinal| {
            let mut frame = frame(ordinal, 1);
            frame.source_dimensions = [5_600, 5_600];
            frame.vectors[0].destination = [2_800, 2_800];
            frame
        };
        let mut sequence = CodecMotionSequence::from_frame(oversized(10)).unwrap();
        sequence.push_contiguous(oversized(11)).unwrap();
        assert_eq!(
            sequence.rasterize_composed(MotionLatticeQuality::High),
            Err(CodecMotionSequenceError::TransientBytes)
        );
    }

    #[test]
    fn gaps_generations_nonadjacent_references_and_unavailable_steps_reject_atomically() {
        let mut sequence = CodecMotionSequence::from_frame(frame(10, 1)).unwrap();
        let before = sequence.clone();
        assert_eq!(
            sequence.push_contiguous(frame(12, 1)),
            Err(CodecMotionSequenceError::OrdinalGap {
                previous: 10,
                next: 12,
            })
        );
        assert_eq!(sequence, before);

        let mut bidirectional = frame(11, 1);
        bidirectional.frame_type = CodecMotionFrameType::Bidirectional;
        assert_eq!(
            sequence.push_contiguous(bidirectional),
            Err(CodecMotionSequenceError::UnsupportedFrameType {
                ordinal: 11,
                frame_type: CodecMotionFrameType::Bidirectional,
            })
        );

        let mut stale = frame(11, 1);
        stale.source_generation = 8;
        let stale_proof = stale.past_reference_proof.as_mut().unwrap();
        stale_proof.reference.source_generation = 8;
        stale_proof.destination.source_generation = 8;
        assert_eq!(
            sequence.push_contiguous(stale),
            Err(CodecMotionSequenceError::GenerationMismatch)
        );

        let mut long_reference = frame(11, 1);
        long_reference.vectors[0].seconds_from_reference = 2.0 / 30.0;
        assert_eq!(
            sequence.push_contiguous(long_reference),
            Err(CodecMotionSequenceError::NonAdjacentReference { ordinal: 11 })
        );

        let mut rejected = frame(11, 1);
        rejected.status = CodecMotionStatus::Rejected(CodecMotionRejectReason::MotionBounds);
        assert_eq!(
            sequence.push_contiguous(rejected),
            Err(CodecMotionSequenceError::UnavailableStep {
                ordinal: 11,
                status: CodecMotionStatus::Rejected(CodecMotionRejectReason::MotionBounds),
            })
        );
        assert_eq!(sequence, before);
    }

    #[test]
    fn frame_and_record_caps_are_explicit_and_do_not_mutate_on_rejection() {
        let mut sequence = CodecMotionSequence::from_frame(frame(1, 1)).unwrap();
        for ordinal in 2..=MAX_CODEC_MOTION_COMPOSITION_FRAMES as u64 {
            sequence.push_contiguous(frame(ordinal, 1)).unwrap();
        }
        let before = sequence.clone();
        assert_eq!(
            sequence.push_contiguous(frame(MAX_CODEC_MOTION_COMPOSITION_FRAMES as u64 + 1, 1)),
            Err(CodecMotionSequenceError::FrameCap)
        );
        assert_eq!(sequence, before);

        let mut oversized = frame(1, 1);
        oversized.vectors = vec![oversized.vectors[0]; MOTION_CODEC_VECTOR_MAX_RECORDS + 1];
        assert_eq!(
            CodecMotionSequence::from_frame(oversized),
            Err(CodecMotionSequenceError::RecordCap)
        );
    }

    #[test]
    fn future_records_are_ignored_only_when_each_step_has_an_adjacent_past_record() {
        let mut mixed = frame(10, 1);
        mixed.vectors.push(CodecMotionVector {
            reference: CodecReferenceDirection::Future,
            ..mixed.vectors[0]
        });
        assert!(CodecMotionSequence::from_frame(mixed).is_ok());

        let mut future_only = frame(10, 1);
        future_only.vectors[0].reference = CodecReferenceDirection::Future;
        assert_eq!(
            CodecMotionSequence::from_frame(future_only),
            Err(CodecMotionSequenceError::UnavailableStep {
                ordinal: 10,
                status: CodecMotionStatus::Available,
            })
        );
    }

    #[test]
    fn product_retags_every_owned_transition_and_exposes_the_exact_latest_facts() {
        let mut sequence = CodecMotionSequence::from_frame(frame(10, 1)).unwrap();
        sequence.push_contiguous(frame(11, 1)).unwrap();
        let mut product = CodecMotionProduct::from(sequence);
        assert_eq!(product.transition_count(), 2);
        assert!((product.elapsed_seconds().unwrap() - 2.0 / 30.0).abs() < 1.0e-6);
        assert_eq!(product.frame_ordinal, 11);
        product.retag_source_generation(99);
        let CodecMotionProduct::Composed(sequence) = product else {
            panic!("two-step product lost its sequence");
        };
        assert!(sequence
            .frames
            .iter()
            .all(|frame| frame.source_generation == 99));
    }
}
