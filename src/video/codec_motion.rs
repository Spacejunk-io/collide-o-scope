//! Bounded, decoder-owned FFmpeg motion-vector metadata.
//!
//! The decoder keeps sparse codec records in donor-local source pixels. It
//! does not rasterize, allocate a field texture, or silently invoke Motion
//! Lattice. Consumers adapt validated records through `crate::motion`.

use std::mem::size_of;
use std::ptr;

use ffmpeg_next as ffmpeg;
use ffmpeg_next::util::frame::side_data;
use ffmpeg_next::util::frame::video::Video as VideoFrame;

use crate::media_safety::ABSOLUTE_MEDIA_MAX_EDGE;
use crate::motion::{
    CodecMotionVector, CodecReferenceDirection, MOTION_ALGORITHM_VERSION,
    MOTION_CODEC_VECTOR_MAX_RECORDS,
};

/// Hard admission bound checked before record count or allocation.
pub const MAX_CODEC_MOTION_SIDE_DATA_BYTES: usize = 16 * 1024 * 1024;
/// Complete owned canonical record allocation for one decoded frame.
pub const MAX_CODEC_MOTION_OWNED_BYTES: usize = 16 * 1024 * 1024;

const MAX_CODEC_BLOCK_PIXELS: u16 = 256;
const MAX_CODEC_REFERENCE_SECONDS: f32 = 10.0;
const DEFAULT_FRAME_DELTA_SECONDS: f32 = 1.0 / 30.0;

/// Honest source of the sparse records. This version never substitutes a
/// computed lattice inside the decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CodecMotionProvenance {
    #[default]
    FfmpegExportMvs,
}

/// Decoded codec picture law, retained even when vector data is unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CodecMotionFrameType {
    Intra,
    Predictive,
    Bidirectional,
    #[default]
    Other,
}

/// Bounded rejection vocabulary; hostile side data never allocates an error
/// string proportional to its contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMotionRejectReason {
    InvalidDimensions,
    InvalidFrameDelta,
    SideDataBytes,
    SideDataLayout,
    RecordCount,
    OwnedBytes,
    Allocation,
    ZeroBlock,
    ZeroMotionScale,
    InvalidReference,
    CoordinateBounds,
    MotionBounds,
    InconsistentCoordinates,
}

/// Availability is distinct from an empty array. Intra frames are known
/// empty; future-only B-frame prediction is intentionally unavailable to the
/// past-reference adapter; malformed side data is explicitly rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMotionStatus {
    Intra,
    Available,
    FutureOnly,
    Unavailable,
    Rejected(CodecMotionRejectReason),
}

/// Immutable sparse codec product paired transactionally with one decoded
/// image. `source_generation` is duplicated deliberately so a host can never
/// pair stale vectors with current pixels by inspecting only this product.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecMotionFrame {
    pub source_dimensions: [u32; 2],
    pub frame_delta_seconds: f32,
    pub source_generation: u64,
    pub frame_ordinal: u64,
    pub algorithm_version: u16,
    pub provenance: CodecMotionProvenance,
    pub frame_type: CodecMotionFrameType,
    pub status: CodecMotionStatus,
    pub vectors: Vec<CodecMotionVector>,
}

impl CodecMotionFrame {
    #[allow(
        dead_code,
        reason = "M4 host source selection consumes this frozen availability predicate"
    )]
    pub fn codec_vectors_available(&self) -> bool {
        self.status == CodecMotionStatus::Available
    }

    pub(crate) fn retag_source_generation(&mut self, source_generation: u64) {
        self.source_generation = source_generation;
    }

    pub(crate) fn from_decoded_frame(
        frame: &VideoFrame,
        source_dimensions: [u32; 2],
        frame_delta_seconds: f32,
        source_generation: u64,
        frame_ordinal: u64,
    ) -> Self {
        let frame_type = CodecMotionFrameType::from_ffmpeg(frame.kind());
        let side_data = frame.side_data(side_data::Type::MotionVectors);
        let side_data_bytes = side_data.as_ref().map(|side_data| side_data.data());
        Self::from_side_data(
            frame_type,
            source_dimensions,
            frame_delta_seconds,
            source_generation,
            frame_ordinal,
            side_data_bytes,
        )
    }

    fn from_side_data(
        frame_type: CodecMotionFrameType,
        source_dimensions: [u32; 2],
        frame_delta_seconds: f32,
        source_generation: u64,
        frame_ordinal: u64,
        side_data: Option<&[u8]>,
    ) -> Self {
        let sanitized_delta = sanitize_frame_delta(frame_delta_seconds);
        let mut product = Self {
            source_dimensions,
            frame_delta_seconds: sanitized_delta,
            source_generation,
            frame_ordinal,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            provenance: CodecMotionProvenance::FfmpegExportMvs,
            frame_type,
            status: CodecMotionStatus::Unavailable,
            vectors: Vec::new(),
        };
        if !valid_source_dimensions(source_dimensions) {
            product.status =
                CodecMotionStatus::Rejected(CodecMotionRejectReason::InvalidDimensions);
            return product;
        }
        if !valid_frame_delta(frame_delta_seconds) {
            product.status =
                CodecMotionStatus::Rejected(CodecMotionRejectReason::InvalidFrameDelta);
            return product;
        }
        if frame_type == CodecMotionFrameType::Intra {
            product.status = CodecMotionStatus::Intra;
            return product;
        }
        let Some(side_data) = side_data else {
            return product;
        };
        match parse_motion_side_data(side_data, source_dimensions, frame_delta_seconds) {
            Ok(vectors) => {
                let observed_past = vectors
                    .iter()
                    .any(|vector| vector.reference == CodecReferenceDirection::Past);
                product.status = if observed_past {
                    CodecMotionStatus::Available
                } else if vectors.is_empty() {
                    CodecMotionStatus::Unavailable
                } else {
                    CodecMotionStatus::FutureOnly
                };
                product.vectors = vectors;
            }
            Err(reason) => product.status = CodecMotionStatus::Rejected(reason),
        }
        product
    }
}

impl CodecMotionFrameType {
    fn from_ffmpeg(kind: ffmpeg::util::picture::Type) -> Self {
        use ffmpeg::util::picture::Type;
        match kind {
            Type::I | Type::SI | Type::BI => Self::Intra,
            Type::P | Type::SP => Self::Predictive,
            Type::B => Self::Bidirectional,
            Type::None | Type::S => Self::Other,
        }
    }
}

fn valid_source_dimensions([width, height]: [u32; 2]) -> bool {
    width > 0 && height > 0 && width <= ABSOLUTE_MEDIA_MAX_EDGE && height <= ABSOLUTE_MEDIA_MAX_EDGE
}

fn valid_frame_delta(value: f32) -> bool {
    value.is_finite() && value > 0.0 && value <= MAX_CODEC_REFERENCE_SECONDS
}

fn sanitize_frame_delta(value: f32) -> f32 {
    if valid_frame_delta(value) {
        value
    } else {
        DEFAULT_FRAME_DELTA_SECONDS
    }
}

fn validate_motion_side_data_len(bytes: usize) -> Result<usize, CodecMotionRejectReason> {
    if bytes > MAX_CODEC_MOTION_SIDE_DATA_BYTES {
        return Err(CodecMotionRejectReason::SideDataBytes);
    }
    let record_bytes = size_of::<ffmpeg::ffi::AVMotionVector>();
    if record_bytes == 0 || !bytes.is_multiple_of(record_bytes) {
        return Err(CodecMotionRejectReason::SideDataLayout);
    }
    let count = bytes / record_bytes;
    if count > MOTION_CODEC_VECTOR_MAX_RECORDS {
        return Err(CodecMotionRejectReason::RecordCount);
    }
    let owned_bytes = count
        .checked_mul(size_of::<CodecMotionVector>())
        .ok_or(CodecMotionRejectReason::OwnedBytes)?;
    if owned_bytes > MAX_CODEC_MOTION_OWNED_BYTES {
        return Err(CodecMotionRejectReason::OwnedBytes);
    }
    Ok(count)
}

fn parse_motion_side_data(
    bytes: &[u8],
    source_dimensions: [u32; 2],
    frame_delta_seconds: f32,
) -> Result<Vec<CodecMotionVector>, CodecMotionRejectReason> {
    let count = validate_motion_side_data_len(bytes.len())?;

    // Validate every record before allocating. One invalid record rejects the
    // complete metadata atomically while the separately decoded pixels remain
    // usable. This also makes the eventual allocation exact-sized.
    for index in 0..count {
        let raw = read_raw_motion_vector(bytes, index);
        canonical_motion_vector(raw, source_dimensions, frame_delta_seconds)?;
    }
    let mut vectors = Vec::new();
    vectors
        .try_reserve_exact(count)
        .map_err(|_| CodecMotionRejectReason::Allocation)?;
    for index in 0..count {
        let raw = read_raw_motion_vector(bytes, index);
        vectors.push(canonical_motion_vector(
            raw,
            source_dimensions,
            frame_delta_seconds,
        )?);
    }
    Ok(vectors)
}

fn read_raw_motion_vector(bytes: &[u8], index: usize) -> ffmpeg::ffi::AVMotionVector {
    let record_bytes = size_of::<ffmpeg::ffi::AVMotionVector>();
    let offset = index * record_bytes;
    // SAFETY: admission proved exact record-size divisibility and `index` is
    // less than the derived count. `read_unaligned` is required because
    // AVFrame side-data bytes do not promise Rust's AVMotionVector alignment.
    unsafe { ptr::read_unaligned(bytes.as_ptr().add(offset).cast()) }
}

fn canonical_motion_vector(
    raw: ffmpeg::ffi::AVMotionVector,
    source_dimensions: [u32; 2],
    frame_delta_seconds: f32,
) -> Result<CodecMotionVector, CodecMotionRejectReason> {
    let block = [u16::from(raw.w), u16::from(raw.h)];
    if block[0] == 0
        || block[1] == 0
        || block[0] > MAX_CODEC_BLOCK_PIXELS
        || block[1] > MAX_CODEC_BLOCK_PIXELS
    {
        return Err(CodecMotionRejectReason::ZeroBlock);
    }
    if raw.motion_scale == 0 {
        return Err(CodecMotionRejectReason::ZeroMotionScale);
    }
    let reference = match raw.source.cmp(&0) {
        std::cmp::Ordering::Less => CodecReferenceDirection::Past,
        std::cmp::Ordering::Greater => CodecReferenceDirection::Future,
        std::cmp::Ordering::Equal => return Err(CodecMotionRejectReason::InvalidReference),
    };
    let seconds_from_reference = raw.source.unsigned_abs() as f64 * f64::from(frame_delta_seconds);
    if !seconds_from_reference.is_finite()
        || seconds_from_reference <= 0.0
        || seconds_from_reference > f64::from(MAX_CODEC_REFERENCE_SECONDS)
    {
        return Err(CodecMotionRejectReason::InvalidReference);
    }

    let coordinate_limit = i64::from(source_dimensions[0].max(source_dimensions[1]))
        + i64::from(MAX_CODEC_BLOCK_PIXELS);
    for coordinate in [raw.src_x, raw.src_y, raw.dst_x, raw.dst_y] {
        if i64::from(coordinate).abs() > coordinate_limit {
            return Err(CodecMotionRejectReason::CoordinateBounds);
        }
    }
    let displacement = [
        f64::from(raw.motion_x) / f64::from(raw.motion_scale),
        f64::from(raw.motion_y) / f64::from(raw.motion_scale),
    ];
    let displacement_limits = [
        f64::from(source_dimensions[0] + u32::from(MAX_CODEC_BLOCK_PIXELS)),
        f64::from(source_dimensions[1] + u32::from(MAX_CODEC_BLOCK_PIXELS)),
    ];
    if displacement
        .iter()
        .zip(displacement_limits)
        .any(|(value, limit)| !value.is_finite() || value.abs() > limit)
    {
        return Err(CodecMotionRejectReason::MotionBounds);
    }
    let expected_source = [
        f64::from(raw.dst_x) + displacement[0],
        f64::from(raw.dst_y) + displacement[1],
    ];
    if [raw.src_x, raw.src_y]
        .into_iter()
        .zip(expected_source)
        .any(|(source, expected)| (f64::from(source) - expected).abs() > 1.0)
    {
        return Err(CodecMotionRejectReason::InconsistentCoordinates);
    }

    // AVMotionVector flags are currently unspecified by FFmpeg. Deliberately
    // do not let opaque bits influence direction, visibility, or confidence.
    let _opaque_flags = raw.flags;
    Ok(CodecMotionVector {
        destination: [i32::from(raw.dst_x), i32::from(raw.dst_y)],
        block,
        motion: [raw.motion_x, raw.motion_y],
        motion_scale: raw.motion_scale,
        seconds_from_reference: seconds_from_reference as f32,
        reference,
        visibility: destination_visibility(
            [i32::from(raw.dst_x), i32::from(raw.dst_y)],
            block,
            source_dimensions,
        ),
    })
}

fn destination_visibility(destination: [i32; 2], block: [u16; 2], dimensions: [u32; 2]) -> f32 {
    let left = i64::from(destination[0]) - i64::from(block[0]) / 2;
    let top = i64::from(destination[1]) - i64::from(block[1]) / 2;
    let right = left + i64::from(block[0]);
    let bottom = top + i64::from(block[1]);
    let clipped_width =
        right.clamp(0, i64::from(dimensions[0])) - left.clamp(0, i64::from(dimensions[0]));
    let clipped_height =
        bottom.clamp(0, i64::from(dimensions[1])) - top.clamp(0, i64::from(dimensions[1]));
    let visible_area = clipped_width.max(0) * clipped_height.max(0);
    let block_area = i64::from(block[0]) * i64::from(block[1]);
    (visible_area as f32 / block_area as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::{rasterize_codec_motion_vectors, MotionLatticeQuality};

    fn raw_vector(
        source: i32,
        destination: [i16; 2],
        motion: [i32; 2],
        scale: u16,
    ) -> ffmpeg::ffi::AVMotionVector {
        // SAFETY: every AVMotionVector member is an integer, so all-zero is a
        // valid initialized representation before explicit fixture fields are
        // assigned. Zeroing also initializes ABI padding copied below.
        let mut vector = unsafe { std::mem::zeroed::<ffmpeg::ffi::AVMotionVector>() };
        vector.source = source;
        vector.w = 8;
        vector.h = 8;
        vector.dst_x = destination[0];
        vector.dst_y = destination[1];
        vector.motion_x = motion[0];
        vector.motion_y = motion[1];
        vector.motion_scale = scale;
        if scale != 0 {
            vector.src_x = i16::try_from(
                i32::from(destination[0]).saturating_add(motion[0] / i32::from(scale)),
            )
            .unwrap_or(destination[0]);
            vector.src_y = i16::try_from(
                i32::from(destination[1]).saturating_add(motion[1] / i32::from(scale)),
            )
            .unwrap_or(destination[1]);
        }
        vector
    }

    fn raw_bytes(vectors: &[ffmpeg::ffi::AVMotionVector]) -> Vec<u8> {
        let byte_len = std::mem::size_of_val(vectors);
        // SAFETY: `raw_vector` initialized every byte, including padding. The
        // returned copy does not outlive the source slice.
        unsafe { std::slice::from_raw_parts(vectors.as_ptr().cast(), byte_len).to_vec() }
    }

    fn parsed(
        frame_type: CodecMotionFrameType,
        vectors: &[ffmpeg::ffi::AVMotionVector],
    ) -> CodecMotionFrame {
        let bytes = raw_bytes(vectors);
        CodecMotionFrame::from_side_data(frame_type, [64, 64], 1.0 / 30.0, 7, 11, Some(&bytes))
    }

    #[test]
    fn known_one_two_and_four_pixel_vectors_keep_ffmpeg_units_and_order() {
        let vectors = [
            raw_vector(-1, [16, 16], [-1, 0], 1),
            raw_vector(-1, [32, 16], [-4, 0], 2),
            raw_vector(-1, [48, 16], [-16, 0], 4),
        ];
        let product = parsed(CodecMotionFrameType::Predictive, &vectors);
        assert_eq!(product.status, CodecMotionStatus::Available);
        assert!(product.codec_vectors_available());
        assert_eq!(product.source_generation, 7);
        assert_eq!(product.frame_ordinal, 11);
        assert_eq!(product.algorithm_version, MOTION_ALGORITHM_VERSION);
        assert_eq!(product.vectors.len(), 3);
        assert_eq!(product.vectors[0].motion, [-1, 0]);
        assert_eq!(product.vectors[1].motion, [-4, 0]);
        assert_eq!(product.vectors[2].motion, [-16, 0]);
        for (vector, expected) in product.vectors.iter().zip([1.0_f32, 2.0, 4.0]) {
            let displacement = -(vector.motion[0] as f32 / f32::from(vector.motion_scale));
            assert_eq!(displacement, expected);
            assert_eq!(vector.reference, CodecReferenceDirection::Past);
            assert_eq!(vector.seconds_from_reference, 1.0 / 30.0);
            assert_eq!(vector.visibility, 1.0);
        }
        assert!(rasterize_codec_motion_vectors(
            [64, 64],
            MotionLatticeQuality::Live,
            &product.vectors,
        )
        .unwrap()
        .is_some());
    }

    #[test]
    fn intra_is_explicitly_empty_and_future_only_is_unavailable() {
        let malformed = raw_vector(-1, [16, 16], [i32::MAX, 0], 0);
        let intra = parsed(CodecMotionFrameType::Intra, &[malformed]);
        assert_eq!(intra.status, CodecMotionStatus::Intra);
        assert!(intra.vectors.is_empty());

        let future = parsed(
            CodecMotionFrameType::Bidirectional,
            &[raw_vector(1, [16, 16], [-2, 0], 1)],
        );
        assert_eq!(future.status, CodecMotionStatus::FutureOnly);
        assert!(!future.codec_vectors_available());
        assert_eq!(future.vectors[0].reference, CodecReferenceDirection::Future);

        let mixed = parsed(
            CodecMotionFrameType::Bidirectional,
            &[
                raw_vector(1, [16, 16], [-2, 0], 1),
                raw_vector(-1, [24, 16], [-2, 0], 1),
            ],
        );
        assert_eq!(mixed.status, CodecMotionStatus::Available);
    }

    #[test]
    fn malformed_admission_rejects_atomically_before_proportional_allocation() {
        assert_eq!(
            validate_motion_side_data_len(MAX_CODEC_MOTION_SIDE_DATA_BYTES + 1),
            Err(CodecMotionRejectReason::SideDataBytes)
        );
        assert_eq!(
            validate_motion_side_data_len(
                (MOTION_CODEC_VECTOR_MAX_RECORDS + 1) * size_of::<ffmpeg::ffi::AVMotionVector>()
            ),
            Err(CodecMotionRejectReason::RecordCount)
        );
        assert_eq!(
            validate_motion_side_data_len(size_of::<ffmpeg::ffi::AVMotionVector>() - 1),
            Err(CodecMotionRejectReason::SideDataLayout)
        );

        for (raw, reason) in [
            (
                raw_vector(-1, [16, 16], [-1, 0], 0),
                CodecMotionRejectReason::ZeroMotionScale,
            ),
            (
                raw_vector(0, [16, 16], [-1, 0], 1),
                CodecMotionRejectReason::InvalidReference,
            ),
            (
                raw_vector(-1, [16, 16], [i32::MAX, 0], 1),
                CodecMotionRejectReason::MotionBounds,
            ),
        ] {
            let product = parsed(CodecMotionFrameType::Predictive, &[raw]);
            assert_eq!(product.status, CodecMotionStatus::Rejected(reason));
            assert!(product.vectors.is_empty());
        }
    }

    #[test]
    fn visibility_is_the_bounded_destination_block_intersection() {
        let product = parsed(
            CodecMotionFrameType::Predictive,
            &[raw_vector(-1, [0, 16], [0, 0], 1)],
        );
        assert_eq!(product.status, CodecMotionStatus::Available);
        assert_eq!(product.vectors[0].visibility, 0.5);
    }

    #[test]
    fn invalid_frame_provenance_is_small_rejected_metadata() {
        let product = CodecMotionFrame::from_side_data(
            CodecMotionFrameType::Predictive,
            [64, 64],
            f32::NAN,
            3,
            4,
            None,
        );
        assert_eq!(
            product.status,
            CodecMotionStatus::Rejected(CodecMotionRejectReason::InvalidFrameDelta)
        );
        assert_eq!(product.frame_delta_seconds, DEFAULT_FRAME_DELTA_SECONDS);
        assert!(product.vectors.is_empty());
    }
}
