use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{context::Input, input_with_interrupt};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::media_safety::{
    validate_safe_dimensions, MediaAllocationPlan, MediaDeviceLimits, MediaReservation,
    MediaSafetyPolicy, MediaSourceKind,
};
use crate::video::indexed::{
    finite_nonnegative, timestamp_to_source_seconds, DecodeWorkError, DecodedVideoFrame,
    FrameMetadata, KeyframeIndex, ReverseFrameCache,
};
use crate::video::{
    AdjacentReferencePolicy, CodecFrameIdentity, CodecMotionFrame, CodecMotionFrameType,
    CodecMotionProduct, CodecMotionSequence, CodecPastReferenceProof, CodecTimeBase,
};

// Compatibility names used by existing callers/tests. Safe mode remains the
// exact UHD-area boundary these constants represented before Expert mode was
// introduced.
#[allow(unused_imports)]
pub use crate::media_safety::{
    ABSOLUTE_MEDIA_MAX_EDGE as MAX_MEDIA_EDGE, SAFE_MEDIA_MAX_PIXELS as MAX_MEDIA_PIXELS,
    SAFE_MEDIA_MAX_RGBA_BYTES as MAX_MEDIA_RGBA_BYTES,
};
/// A valid inter-frame stream should never need anywhere near this many
/// packets to produce one image. The cap makes corrupt packet-only streams
/// fail deterministically even before EOF.
// A valid decoder may buffer a modest reorder window, but thousands of video
// packets without one frame is already malformed/unsupported input. Keep the
// bound practical enough that opening such a file cannot pin the UI for an
// unbounded-feeling scan.
const MAX_PACKETS_WITHOUT_FRAME: u32 = 4_096;
/// Local files should open promptly. The interrupt callback also bounds
/// FFmpeg stream probing if a malformed file stalls demuxer I/O.
const MEDIA_IO_TIMEOUT: Duration = Duration::from_secs(5);
/// A hostile keyframe table cannot make one absolute selection decode an
/// unbounded number of frames before returning control to the command worker.
const MAX_SEEK_DECODE_FRAMES: u32 = 4_096;

/// FFmpeg only attaches `AV_FRAME_DATA_MOTION_VECTORS` when this flag is set
/// before `avcodec_open2`. Keep the narrow unsafe ABI write in one place and
/// call it for initial open, dimension probe, and every EOF reopen.
fn enable_export_motion_vectors(context: &mut ffmpeg::codec::context::Context) {
    // SAFETY: `Context` owns a live, uniquely borrowed AVCodecContext. The
    // flag is part of FFmpeg's public ABI and is written before decoder open.
    unsafe {
        (*context.as_mut_ptr()).flags2 |= ffmpeg::ffi::AV_CODEC_FLAG2_EXPORT_MVS;
    }
}

/// Where the live FFmpeg decoder actually sits, as opposed to which picture
/// was last *selected*. Only a real decode advances it; a seek or a reopen
/// clears it. A reverse cache hit deliberately leaves it alone, because a
/// cache hit serves pixels without moving the decoder at all.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StreamPosition {
    source_seconds: f64,
    pts: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderSeekStats {
    pub index_builds: u64,
    pub seek_calls: u64,
    pub reverse_cache_hits: u64,
    pub reopen_calls: u64,
    /// Kept explicit as a regression seam: indexed selection must never fall
    /// back to reopening and scanning from frame zero.
    pub scans_from_zero: u64,
}

pub(super) struct KeyframeIndexBuildRequest {
    path: String,
    cancel: Arc<AtomicBool>,
    stream_time_base_seconds: f64,
    stream_start_pts: i64,
}

impl KeyframeIndexBuildRequest {
    pub(super) fn build(self) -> Result<KeyframeIndex, String> {
        let mut input = open_input(&self.path, self.cancel.clone(), "index")?;
        let stream_index = input
            .streams()
            .best(Type::Video)
            .ok_or("No video stream while indexing")?
            .index();
        KeyframeIndex::scan(
            &mut input,
            stream_index,
            self.stream_time_base_seconds,
            self.stream_start_pts,
            &self.cancel,
        )
    }
}

pub struct VideoDecoder {
    path: String,
    input_ctx: Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    scaler: ScalerContext,
    pub width: u32,
    pub height: u32,
    /// Source stream's average frame rate. Falls back to 30 when the
    /// container does not provide a usable value.
    pub fps: f32,
    stream_time_base_seconds: f64,
    stream_start_pts: i64,
    duration_seconds: f64,
    last_pts: Option<i64>,
    last_source_seconds: f64,
    /// The live decoder's own position. This is deliberately **not**
    /// `last_source_seconds`: a reverse cache hit adopts that field from
    /// cached pixels while the decoder stays exactly where it was, so only a
    /// decode may write this one.
    stream_position: Option<StreamPosition>,
    frame_count: u64,
    total_frames: u64,
    /// Number of times EOF was reached and the input was successfully
    /// reopened. This never resets during the decoder's lifetime.
    loop_generation: u64,
    cancel: Arc<AtomicBool>,
    /// Retains any above-UHD Expert working-set reservation for the complete
    /// decoder lifetime, including EOF reopens.
    _media_reservation: MediaReservation,
    media_plan: MediaAllocationPlan,
    keyframe_index: KeyframeIndex,
    reverse_cache: ReverseFrameCache,
    seek_stats: DecoderSeekStats,
    /// FFmpeg's AVMotionVector side data does not identify the exact reference
    /// picture. MPEG-4 Part 2 P/B VOPs have a bounded previous-anchor law that
    /// lets this decoder derive the past interval from decoded timestamps.
    /// Other codecs remain conservatively unavailable to codec-vector motion.
    codec_motion_previous_anchor_law: bool,
    codec_time_base: Option<CodecTimeBase>,
    codec_motion_generation: Option<u64>,
    codec_previous_output: Option<CodecFrameIdentity>,
    codec_previous_anchor: Option<CodecFrameIdentity>,
}

impl VideoDecoder {
    /// Safe-policy compatibility constructor retained for embedders and tests.
    #[allow(dead_code)]
    pub fn open(path: &str) -> Result<Self, String> {
        Self::open_with_cancel(path, Arc::new(AtomicBool::new(false)))
    }

    /// Open a decoder with a cooperative cancellation flag shared by the
    /// caller. FFmpeg's interrupt callback observes both cancellation and a
    /// fixed I/O deadline during open and every later packet read/reopen.
    /// Cancellation-aware Safe-policy compatibility constructor.
    #[allow(dead_code)]
    pub fn open_with_cancel(path: &str, cancel: Arc<AtomicBool>) -> Result<Self, String> {
        let policy = MediaSafetyPolicy::safe();
        Self::open_with_cancel_and_media_policy(path, cancel, &policy, MediaDeviceLimits::none())
    }

    /// Inspect codec dimensions under the same admission policy without
    /// constructing a scaler or allocating a decoded-frame mailbox. Startup
    /// uses this to choose a preview size; actual layer construction still
    /// reserves the complete working set below.
    pub fn probe_dimensions_with_media_policy(
        path: &str,
        media_policy: &MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Result<MediaAllocationPlan, String> {
        ffmpeg::init().map_err(|error| format!("ffmpeg init failed: {error}"))?;
        let input_ctx = open_input(path, Arc::new(AtomicBool::new(false)), "dimension probe")?;
        let stream = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream found")?;
        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|error| format!("Codec params: {error}"))?;
        enable_export_motion_vectors(&mut codec_ctx);
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|error| format!("Decoder: {error}"))?;
        media_policy
            .plan(
                MediaSourceKind::Video,
                decoder.width(),
                decoder.height(),
                device_limits,
            )
            .map_err(|error| format!("video source rejected: {error}"))
    }

    /// Open using an explicit host-local media policy and detected device
    /// limits. Patches never construct or alter this policy.
    #[allow(dead_code)]
    pub fn open_with_media_policy(
        path: &str,
        media_policy: &MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Result<Self, String> {
        Self::open_with_cancel_and_media_policy(
            path,
            Arc::new(AtomicBool::new(false)),
            media_policy,
            device_limits,
        )
    }

    /// Cancellation-aware form used by the threaded decoder worker.
    pub fn open_with_cancel_and_media_policy(
        path: &str,
        cancel: Arc<AtomicBool>,
        media_policy: &MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))?;

        let input_ctx = open_input(path, cancel.clone(), "open")?;

        let stream = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream found")?;
        let stream_index = stream.index();
        let codec_time_base = CodecTimeBase::new(
            stream.time_base().numerator(),
            stream.time_base().denominator(),
        );

        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        enable_export_motion_vectors(&mut codec_ctx);
        let codec_id = codec_ctx.id();
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder: {e}"))?;
        let codec_motion_previous_anchor_law =
            codec_has_previous_anchor_motion(codec_id, decoder.profile(), decoder.has_b_frames());

        let width = decoder.width();
        let height = decoder.height();
        let media_reservation = media_policy
            .reserve_source(MediaSourceKind::Video, width, height, device_limits)
            .map_err(|error| format!("video source rejected: {error}"))?;
        let media_plan = media_reservation.plan().clone();
        let avg_fps = f64::from(stream.avg_frame_rate());
        let fps = if avg_fps.is_finite() && avg_fps > 0.0 {
            avg_fps as f32
        } else {
            30.0
        };
        let stream_time_base_seconds = f64::from(stream.time_base());
        let stream_time_base_seconds =
            if stream_time_base_seconds.is_finite() && stream_time_base_seconds > 0.0 {
                stream_time_base_seconds
            } else {
                1.0 / f64::from(fps)
            };
        let raw_start_pts = stream.start_time();
        let stream_start_pts = if raw_start_pts == ffmpeg::ffi::AV_NOPTS_VALUE {
            0
        } else {
            raw_start_pts
        };
        let stream_duration_seconds =
            if stream.duration() > 0 && stream.duration() != ffmpeg::ffi::AV_NOPTS_VALUE {
                stream.duration() as f64 * stream_time_base_seconds
            } else {
                0.0
            };

        // Estimate total frames from stream duration and frame rate
        let total_frames = {
            let frames = stream.frames() as u64;
            if frames > 0 {
                frames
            } else {
                // Fallback: compute from duration and fps
                let duration_secs =
                    input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
                let estimated = (duration_secs * fps as f64) as u64;
                estimated.max(1)
            }
        };
        let container_duration_seconds =
            input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
        let duration_seconds =
            if stream_duration_seconds.is_finite() && stream_duration_seconds > 0.0 {
                stream_duration_seconds
            } else {
                finite_nonnegative(container_duration_seconds)
            };

        let scaler = ScalerContext::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGBA,
            width,
            height,
            Flags::BILINEAR,
        )
        .map_err(|e| format!("Scaler: {e}"))?;

        Ok(Self {
            path: path.to_string(),
            input_ctx,
            stream_index,
            decoder,
            scaler,
            width,
            height,
            fps,
            stream_time_base_seconds,
            stream_start_pts,
            duration_seconds,
            last_pts: None,
            last_source_seconds: 0.0,
            stream_position: None,
            frame_count: 0,
            total_frames,
            loop_generation: 0,
            cancel,
            _media_reservation: media_reservation,
            media_plan,
            keyframe_index: KeyframeIndex::fallback(stream_start_pts)?,
            reverse_cache: ReverseFrameCache::with_ledger(
                usize::try_from(media_policy.reverse_cache_bytes_per_decoder())
                    .unwrap_or(usize::MAX),
                super::indexed::MAX_REVERSE_CACHE_FRAMES,
                media_policy.reverse_cache_ledger(),
            ),
            seek_stats: DecoderSeekStats::default(),
            codec_motion_previous_anchor_law,
            codec_time_base,
            codec_motion_generation: None,
            codec_previous_output: None,
            codec_previous_anchor: None,
        })
    }

    /// Metadata-rich sequential decode used by transport-aware workers.
    pub fn next_timed_frame_result(
        &mut self,
        source_generation: u64,
    ) -> Result<DecodedVideoFrame, String> {
        self.next_timed_frame_interruptible(source_generation, || true)
            .map_err(|error| error.to_string())
    }

    /// Generation-aware sequential decode. `is_current` is checked between
    /// every demuxed packet and decoded frame so superseded work yields
    /// promptly without poisoning decoder health.
    pub fn next_timed_frame_interruptible<F>(
        &mut self,
        source_generation: u64,
        mut is_current: F,
    ) -> Result<DecodedVideoFrame, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        let mut packets_without_frame = 0u32;
        loop {
            self.check_work_current(&mut is_current)?;
            // Try to receive already-decoded frames first
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                self.frame_count += 1;
                return self
                    .finish_decoded_frame(&decoded, source_generation)
                    .map_err(DecodeWorkError::Failed);
            }

            // Feed more packets to the decoder
            match self
                .next_video_packet_interruptible(&mut packets_without_frame, &mut is_current)?
            {
                Some(packet) => {
                    self.decoder.send_packet(&packet).map_err(|e| {
                        DecodeWorkError::Failed(format!(
                            "Failed to submit packet while decoding {}: {e}",
                            self.path
                        ))
                    })?;
                    self.check_work_current(&mut is_current)?;
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return self
                            .finish_decoded_frame(&decoded, source_generation)
                            .map_err(DecodeWorkError::Failed);
                    }
                }
                None => {
                    // EOF — flush decoder then loop the file
                    self.decoder.send_eof().ok();
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return self
                            .finish_decoded_frame(&decoded, source_generation)
                            .map_err(DecodeWorkError::Failed);
                    }
                    // One zero-frame pass is malformed/unsupported input, not
                    // a looping video. Reopening it forever would wedge layer
                    // startup and offline export cancellation.
                    require_decoded_frame_before_loop(self.frame_count, &self.path)
                        .map_err(DecodeWorkError::Failed)?;
                    // This call already decoded at least one frame before
                    // reaching EOF, so reopen for normal loop playback.
                    self.check_work_current(&mut is_current)?;
                    self.reopen().map_err(DecodeWorkError::Failed)?;
                    self.check_work_current(&mut is_current)?;
                    packets_without_frame = 0;
                }
            }
        }
    }

    /// Build a bounded keyframe index using a second demuxer. Threaded live
    /// playback invokes this after publishing startup metadata, so packet
    /// scanning never occurs in the render callback or delays the seed frame.
    #[allow(dead_code)]
    pub fn build_keyframe_index(&mut self) -> Result<usize, String> {
        let index = self.keyframe_index_build_request().build()?;
        let len = index.len();
        self.install_keyframe_index(index);
        Ok(len)
    }

    pub(super) fn keyframe_index_build_request(&self) -> KeyframeIndexBuildRequest {
        KeyframeIndexBuildRequest {
            path: self.path.clone(),
            cancel: self.cancel.clone(),
            stream_time_base_seconds: self.stream_time_base_seconds,
            stream_start_pts: self.stream_start_pts,
        }
    }

    pub(super) fn install_keyframe_index(&mut self, index: KeyframeIndex) {
        self.keyframe_index = index;
        self.seek_stats.index_builds = self.seek_stats.index_builds.saturating_add(1);
    }

    /// Select the frame at an absolute source time. FFmpeg seeks to the
    /// indexed preceding keyframe, flushes codec reorder state, and decodes
    /// forward. A recent reverse selection may be fulfilled by the bounded
    /// decoded-frame cache without any demuxer reopen or scan from zero.
    #[allow(dead_code)]
    pub fn seek_decode(&mut self, target_seconds: f64) -> Result<DecodedVideoFrame, String> {
        self.seek_decode_for_generation(target_seconds, 0)
    }

    #[allow(dead_code)]
    pub fn seek_decode_for_generation(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
    ) -> Result<DecodedVideoFrame, String> {
        self.seek_decode_interruptible(target_seconds, source_generation, || true)
            .map_err(|error| error.to_string())
    }

    pub fn seek_decode_interruptible<F>(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
        is_current: F,
    ) -> Result<DecodedVideoFrame, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        self.seek_decode_internal(target_seconds, source_generation, None, false, is_current)
    }

    /// Select one absolute frame while retaining every adjacent codec-motion
    /// transition after the previously accepted source image. Passing `None`
    /// is a source cut: the decoded pixels remain valid, but codec motion is
    /// intentionally unavailable for that first image.
    pub fn seek_decode_after_for_generation(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
        previous_accepted_identity: Option<CodecFrameIdentity>,
    ) -> Result<DecodedVideoFrame, String> {
        self.seek_decode_after_interruptible(
            target_seconds,
            source_generation,
            previous_accepted_identity,
            || true,
        )
        .map_err(|error| error.to_string())
    }

    pub fn seek_decode_after_interruptible<F>(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
        previous_accepted_identity: Option<CodecFrameIdentity>,
        is_current: F,
    ) -> Result<DecodedVideoFrame, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        self.seek_decode_internal(
            target_seconds,
            source_generation,
            previous_accepted_identity,
            true,
            is_current,
        )
    }

    fn seek_decode_internal<F>(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
        previous_accepted_identity: Option<CodecFrameIdentity>,
        compose_skipped_motion: bool,
        mut is_current: F,
    ) -> Result<DecodedVideoFrame, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        if !target_seconds.is_finite() {
            return Err(DecodeWorkError::Failed(format!(
                "cannot seek {} to non-finite source time {target_seconds}",
                self.path
            )));
        }
        self.check_work_current(&mut is_current)?;
        let target_seconds = if self.duration_seconds > 0.0 {
            target_seconds.clamp(0.0, self.duration_seconds)
        } else {
            target_seconds.max(0.0)
        };
        let frame_period = 1.0 / f64::from(self.fps.max(1.0));
        let reverse_request =
            is_strict_reverse_request(target_seconds, self.last_source_seconds, frame_period);
        if reverse_request {
            if let Some(frame) = self
                .reverse_cache
                .near_at_or_before(target_seconds, source_generation, frame_period * 1.5)
                .map_err(DecodeWorkError::Failed)?
            {
                self.check_work_current(&mut is_current)?;
                self.seek_stats.reverse_cache_hits =
                    self.seek_stats.reverse_cache_hits.saturating_add(1);
                self.adopt_selected_metadata(frame.metadata);
                // Cached pixels are exact, but the live FFmpeg decoder is not
                // positioned at this cached picture. No subsequent decode may
                // inherit its prior anchor state as if the cache hit decoded.
                self.reset_codec_motion_reference_state();
                return Ok(frame);
            }
        }

        let previous_accepted_identity = previous_accepted_identity
            .filter(|identity| identity.source_generation == source_generation);

        let keyframe = self.keyframe_index.preceding(target_seconds);
        let mut continue_forward = can_continue_forward(
            self.stream_position,
            target_seconds,
            keyframe.source_seconds,
            frame_period,
            previous_accepted_identity,
        );
        loop {
            if !continue_forward {
                self.seek_to_preceding_keyframe(keyframe.pts)
                    .map_err(DecodeWorkError::Failed)?;
                self.check_work_current(&mut is_current)?;
                self.frame_count = (keyframe.source_seconds * f64::from(self.fps))
                    .floor()
                    .max(0.0) as u64;
            }
            match self.walk_to_target(
                target_seconds,
                source_generation,
                previous_accepted_identity,
                compose_skipped_motion,
                frame_period,
                &mut is_current,
            ) {
                Ok(frame) => return Ok(frame),
                // A forward walk can end holding nothing where the seek walk
                // would have re-decoded the group of pictures and therefore
                // still held a terminal frame — a decoder already sitting on
                // the last picture decodes straight into EOF. Continuing
                // forward is an optimization and is never permitted to be
                // worse than the walk it replaced, so redo it the old way.
                // `Superseded` is deliberately not retried: that work was
                // abandoned on purpose.
                Err(DecodeWorkError::Failed(_)) if continue_forward => {
                    continue_forward = false;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Decode forward from wherever the decoder currently sits until the
    /// picture that satisfies the target is reached.
    fn walk_to_target<F>(
        &mut self,
        target_seconds: f64,
        source_generation: u64,
        previous_accepted_identity: Option<CodecFrameIdentity>,
        compose_skipped_motion: bool,
        frame_period: f64,
        is_current: &mut F,
    ) -> Result<DecodedVideoFrame, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        let mut motion_sequence = None;
        let mut motion_sequence_rejected = false;
        let mut newest = None;
        let mut reached_eof = false;
        for _ in 0..MAX_SEEK_DECODE_FRAMES {
            self.check_work_current(is_current)?;
            let Some(mut frame) =
                self.next_frame_without_loop_interruptible(source_generation, is_current)?
            else {
                reached_eof = true;
                break;
            };
            if compose_skipped_motion {
                append_codec_motion_after(
                    &mut motion_sequence,
                    &mut motion_sequence_rejected,
                    &frame,
                    previous_accepted_identity,
                );
            }
            let reached = frame.metadata.source_seconds + frame_period * 0.5 >= target_seconds;
            if reached {
                if compose_skipped_motion {
                    frame.codec_motion =
                        accepted_codec_motion_sequence(motion_sequence, motion_sequence_rejected);
                }
                return Ok(frame);
            }
            newest = Some(frame);
        }
        if compose_skipped_motion {
            if let Some(frame) = newest.as_mut() {
                frame.codec_motion =
                    accepted_codec_motion_sequence(motion_sequence, motion_sequence_rejected);
            }
        }
        finish_unreached_seek(
            newest,
            reached_eof,
            target_seconds,
            self.duration_seconds,
            frame_period,
            &self.path,
        )
    }

    /// Returns loop progress as 0.0..1.0
    pub fn progress(&self) -> f32 {
        if self.duration_seconds > 0.0 {
            return (self.last_source_seconds / self.duration_seconds).clamp(0.0, 1.0) as f32;
        }
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.frame_count % self.total_frames) as f32 / self.total_frames as f32
    }

    /// Cumulative count of successful EOF reopens.
    pub fn loop_generation(&self) -> u64 {
        self.loop_generation
    }

    /// Checked allocation plan retained for status/reporting integrations.
    pub fn media_allocation_plan(&self) -> &MediaAllocationPlan {
        &self.media_plan
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }

    #[allow(dead_code)]
    pub fn seek_stats(&self) -> DecoderSeekStats {
        self.seek_stats
    }

    #[allow(dead_code)]
    pub fn reverse_cache_usage(&self) -> (usize, usize) {
        (self.reverse_cache.len(), self.reverse_cache.bytes())
    }

    fn next_frame_without_loop_interruptible<F>(
        &mut self,
        source_generation: u64,
        is_current: &mut F,
    ) -> Result<Option<DecodedVideoFrame>, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        let mut packets_without_frame = 0u32;
        loop {
            self.check_work_current(is_current)?;
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                self.frame_count = self.frame_count.saturating_add(1);
                return self
                    .finish_decoded_frame(&decoded, source_generation)
                    .map(Some)
                    .map_err(DecodeWorkError::Failed);
            }
            match self.next_video_packet_interruptible(&mut packets_without_frame, is_current)? {
                Some(packet) => {
                    self.decoder.send_packet(&packet).map_err(|error| {
                        DecodeWorkError::Failed(format!(
                            "Failed to submit packet while seeking {}: {error}",
                            self.path
                        ))
                    })?;
                }
                None => {
                    self.decoder.send_eof().ok();
                    let mut drained = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut drained).is_ok() {
                        self.frame_count = self.frame_count.saturating_add(1);
                        return self
                            .finish_decoded_frame(&drained, source_generation)
                            .map(Some)
                            .map_err(DecodeWorkError::Failed);
                    }
                    return Ok(None);
                }
            }
        }
    }

    fn seek_to_preceding_keyframe(&mut self, pts: i64) -> Result<(), String> {
        self.check_cancelled()?;
        // Repositioning invalidates the live position on every exit path,
        // including a failed seek: after this call the decoder is somewhere
        // this type can no longer name until it decodes again.
        self.stream_position = None;
        // SAFETY: `input_ctx` owns a live AVFormatContext for the complete
        // call, the stream index was obtained from that context, and FFmpeg
        // does not retain any Rust pointer. Restricting max_ts to `pts` plus
        // BACKWARD prevents selection of a later keyframe.
        let result = unsafe {
            ffmpeg::ffi::avformat_seek_file(
                self.input_ctx.as_mut_ptr(),
                i32::try_from(self.stream_index)
                    .map_err(|_| "video stream index does not fit FFmpeg".to_string())?,
                i64::MIN,
                pts,
                pts,
                ffmpeg::ffi::AVSEEK_FLAG_BACKWARD,
            )
        };
        if result < 0 {
            return Err(format!(
                "FFmpeg could not seek {} to preceding keyframe PTS {pts}: {}",
                self.path,
                ffmpeg::Error::from(result)
            ));
        }
        self.decoder.flush();
        self.reset_codec_motion_reference_state();
        self.seek_stats.seek_calls = self.seek_stats.seek_calls.saturating_add(1);
        Ok(())
    }

    fn finish_decoded_frame(
        &mut self,
        frame: &VideoFrame,
        source_generation: u64,
    ) -> Result<DecodedVideoFrame, String> {
        let pts = frame.timestamp().or_else(|| frame.pts());
        let fallback_seconds = self.frame_count.saturating_sub(1) as f64 / f64::from(self.fps);
        let source_seconds = pts.map_or(fallback_seconds, |timestamp| {
            timestamp_to_source_seconds(
                timestamp,
                self.stream_start_pts,
                self.stream_time_base_seconds,
            )
        });
        let identity = pts.map(|pts| CodecFrameIdentity {
            source_generation,
            pts,
            presentation_ordinal: self.frame_count.saturating_sub(1),
        });
        let metadata = FrameMetadata::sanitized(
            source_generation,
            pts,
            source_seconds,
            self.duration_seconds,
        )
        .with_codec_identity(identity);
        let rgba = self.scale_frame(frame)?;
        if self.codec_motion_generation != Some(source_generation) {
            self.reset_codec_motion_reference_state();
            self.codec_motion_generation = Some(source_generation);
        }
        let frame_type = CodecMotionFrameType::from_ffmpeg(frame.kind());
        let frame_delta_seconds = self
            .codec_time_base
            .and_then(|time_base| {
                exact_identity_elapsed(self.codec_previous_output, identity, time_base)
            })
            .unwrap_or_else(|| 1.0 / self.fps.max(1.0));
        let past_reference_proof = if self.codec_motion_previous_anchor_law
            && frame_type == CodecMotionFrameType::Predictive
            && !frame.is_interlaced()
            && !frame.is_corrupt()
            && self.codec_previous_anchor == self.codec_previous_output
        {
            exact_adjacent_reference_proof(
                self.codec_previous_anchor,
                identity,
                self.codec_time_base,
            )
        } else {
            None
        };
        let codec_motion = Some(
            CodecMotionFrame::from_decoded_frame(
                frame,
                [self.width, self.height],
                frame_delta_seconds,
                past_reference_proof,
                source_generation,
                self.frame_count.saturating_sub(1),
            )
            .into(),
        );
        self.codec_previous_output = identity;
        if !frame.is_interlaced()
            && !frame.is_corrupt()
            && matches!(
                frame_type,
                CodecMotionFrameType::Intra | CodecMotionFrameType::Predictive
            )
        {
            self.codec_previous_anchor = identity;
        } else {
            // A corrupt/interlaced output cannot become the proven reference
            // for the next progressive P picture merely because it occupied
            // the immediately preceding presentation slot.
            self.codec_previous_anchor = None;
        }
        let decoded = DecodedVideoFrame {
            rgba,
            metadata,
            codec_motion,
        };
        self.adopt_selected_metadata(metadata);
        // A real decode is the only thing that moves the live decoder, so this
        // is the one place the stream position may advance.
        self.stream_position = Some(StreamPosition {
            source_seconds: metadata.source_seconds,
            pts: metadata.pts,
        });
        if let Err(error) = self.reverse_cache.insert(&decoded) {
            // Reverse acceleration is optional. A cache allocation failure
            // cannot poison forward playback or source selection.
            self.reverse_cache.clear();
            log::warn!("{error}");
        }
        Ok(decoded)
    }

    fn adopt_selected_metadata(&mut self, metadata: FrameMetadata) {
        self.last_pts = metadata.pts;
        self.last_source_seconds = metadata.source_seconds;
    }

    fn reset_codec_motion_reference_state(&mut self) {
        self.codec_motion_generation = None;
        self.codec_previous_output = None;
        self.codec_previous_anchor = None;
    }

    fn scale_frame(&mut self, frame: &VideoFrame) -> Result<Vec<u8>, String> {
        let mut rgb_frame = VideoFrame::empty();
        self.scaler.run(frame, &mut rgb_frame).map_err(|e| {
            format!(
                "Failed to scale decoded frame from {} ({}x{} to RGBA): {e}",
                self.path, self.width, self.height
            )
        })?;
        repack_rgba_plane(
            rgb_frame.data(0),
            rgb_frame.stride(0),
            self.width,
            self.height,
        )
    }

    fn next_video_packet_interruptible<F>(
        &mut self,
        packets_without_frame: &mut u32,
        is_current: &mut F,
    ) -> Result<Option<ffmpeg::Packet>, DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        for (stream, packet) in self.input_ctx.packets() {
            if !is_current() {
                return Err(DecodeWorkError::Superseded);
            }
            if self.cancel.load(Ordering::Acquire) {
                return Err(DecodeWorkError::Failed(format!(
                    "video decode cancelled for {}",
                    self.path
                )));
            }
            if stream.index() == self.stream_index {
                *packets_without_frame =
                    count_packet_without_frame(*packets_without_frame, &self.path)
                        .map_err(DecodeWorkError::Failed)?;
                return Ok(Some(packet));
            }
        }
        Ok(None)
    }

    fn reopen(&mut self) -> Result<(), String> {
        self.input_ctx = open_input(&self.path, self.cancel.clone(), "reopen")?;

        let stream = self
            .input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream on reopen")?;

        let stream_index = stream.index();
        let codec_time_base = CodecTimeBase::new(
            stream.time_base().numerator(),
            stream.time_base().denominator(),
        );
        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        enable_export_motion_vectors(&mut codec_ctx);
        let codec_id = codec_ctx.id();
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;
        let codec_motion_previous_anchor_law =
            codec_has_previous_anchor_motion(codec_id, decoder.profile(), decoder.has_b_frames());
        if decoder.width() != self.width || decoder.height() != self.height {
            return Err(format!(
                "video dimensions changed while reopening {}: expected {}x{}, got {}x{}",
                self.path,
                self.width,
                self.height,
                decoder.width(),
                decoder.height()
            ));
        }
        self.stream_index = stream_index;
        self.decoder = decoder;
        self.codec_motion_previous_anchor_law = codec_motion_previous_anchor_law;
        self.codec_time_base = codec_time_base;

        self.frame_count = 0;
        self.loop_generation = next_loop_generation(self.loop_generation);
        self.last_pts = None;
        self.last_source_seconds = 0.0;
        self.stream_position = None;
        self.reset_codec_motion_reference_state();
        self.seek_stats.reopen_calls = self.seek_stats.reopen_calls.saturating_add(1);

        Ok(())
    }

    fn check_cancelled(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Acquire) {
            Err(format!("video decode cancelled for {}", self.path))
        } else {
            Ok(())
        }
    }

    fn check_work_current<F>(&self, is_current: &mut F) -> Result<(), DecodeWorkError>
    where
        F: FnMut() -> bool,
    {
        if !is_current() {
            Err(DecodeWorkError::Superseded)
        } else {
            self.check_cancelled().map_err(DecodeWorkError::Failed)
        }
    }
}

fn is_strict_reverse_request(
    target_seconds: f64,
    last_source_seconds: f64,
    frame_period: f64,
) -> bool {
    target_seconds.is_finite()
        && last_source_seconds.is_finite()
        && frame_period.is_finite()
        && target_seconds >= 0.0
        && last_source_seconds >= 0.0
        && frame_period > 0.0
        && target_seconds + frame_period * 0.25 < last_source_seconds
}

/// Narrow codec whitelist whose bitstream law identifies every past-directed
/// exported vector on a P VOP with the immediately previous I/P anchor. Only
/// progressive, uncorrupted destinations are admitted at frame decode. The
/// exact interval still comes from integer presentation timestamps; the ABI's
/// `AVMotionVector.source` magnitude is never interpreted as distance.
fn codec_has_previous_anchor_motion(
    codec_id: ffmpeg::codec::Id,
    profile: ffmpeg::codec::Profile,
    has_b_frames: bool,
) -> bool {
    matches!(
        (codec_id, profile, has_b_frames),
        (
            ffmpeg::codec::Id::MPEG4,
            ffmpeg::codec::Profile::MPEG4(ffmpeg::codec::profile::MPEG4::Simple),
            false
        )
    )
}

fn exact_identity_elapsed(
    previous: Option<CodecFrameIdentity>,
    current: Option<CodecFrameIdentity>,
    time_base: CodecTimeBase,
) -> Option<f32> {
    let previous = previous?;
    let current = current?;
    if previous.source_generation != current.source_generation
        || current.presentation_ordinal != previous.presentation_ordinal.saturating_add(1)
    {
        return None;
    }
    time_base.elapsed_seconds(current.pts.checked_sub(previous.pts)?)
}

fn exact_adjacent_reference_proof(
    reference: Option<CodecFrameIdentity>,
    destination: Option<CodecFrameIdentity>,
    time_base: Option<CodecTimeBase>,
) -> Option<CodecPastReferenceProof> {
    let reference = reference?;
    let destination = destination?;
    let time_base = time_base?;
    let elapsed_ticks = destination.pts.checked_sub(reference.pts)?;
    let proof = CodecPastReferenceProof {
        policy: AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
        reference,
        destination,
        elapsed_ticks,
        time_base,
    };
    proof.elapsed_seconds().map(|_| proof)
}

/// Whether the requested picture can be reached by decoding forward from where
/// the live decoder already is, instead of seeking back to the preceding
/// keyframe and walking the whole group of pictures again.
///
/// Without this, every forward advance — including an ordinary one-frame step —
/// costs a seek, a flush, and a decode of the entire GOP. The cost is therefore
/// O(GOP), which for a 240-frame 1080p GOP is over a second of work per
/// displayed frame and reads to an operator as a frozen picture.
///
/// Two conditions make the forward walk return *provably* the same picture the
/// seek walk would return:
///
/// * no keyframe lies between the live position and the target, so a seek would
///   land at or behind where the decoder already is and the seek walk would
///   decode through this position anyway; and
/// * the target is far enough ahead that the picture at the live position fails
///   the walk's own `reached` test — `source_seconds + frame_period * 0.5 >=
///   target_seconds`. Every picture at or before the live position then fails
///   it too, because presentation times increase, so the first picture that
///   satisfies it is the same one in both walks.
///
/// The previously accepted identity is the last picture the *renderer
/// consumed*, while the position is the last one the *decoder produced*. Those
/// diverge whenever a decode is superseded, and requiring them to be equal
/// builds a trap: the first decode of a long group of pictures is slow enough
/// to be superseded, after which the fast path can never engage again and every
/// frame seeks forever. So the position is required only to be at or after the
/// accepted picture, never behind it.
///
/// When it is strictly after, the transitions between the two belong to frames
/// the renderer discarded, and continuing forward does not collect them. That
/// is safe rather than merely tolerable: `append_codec_motion_after` requires
/// the first transition it appends to *prove* its past reference is the
/// accepted identity, so a broken chain is rejected whole and the frame simply
/// carries no composed sequence. Codec motion can therefore be briefly
/// unavailable while the pipeline recovers, but it can never be wrong — and
/// once decodes stop being superseded the two agree again and the sequence is
/// composed exactly as before.
fn can_continue_forward(
    position: Option<StreamPosition>,
    target_seconds: f64,
    keyframe_seconds: f64,
    frame_period: f64,
    previous_accepted_identity: Option<CodecFrameIdentity>,
) -> bool {
    let Some(position) = position else {
        return false;
    };
    if !position.source_seconds.is_finite() || position.source_seconds < keyframe_seconds {
        return false;
    }
    if target_seconds <= position.source_seconds + frame_period * 0.5 {
        return false;
    }
    match previous_accepted_identity {
        Some(identity) => position.pts.is_some_and(|pts| pts >= identity.pts),
        None => true,
    }
}

fn append_codec_motion_after(
    sequence: &mut Option<CodecMotionSequence>,
    rejected: &mut bool,
    frame: &DecodedVideoFrame,
    previous_accepted_identity: Option<CodecFrameIdentity>,
) {
    let Some(previous_identity) = previous_accepted_identity else {
        return;
    };
    if *rejected {
        return;
    }
    let Some(product) = frame.codec_motion.as_ref() else {
        *sequence = None;
        *rejected = true;
        return;
    };
    let transition = product.latest().clone();
    if transition.frame_ordinal <= previous_identity.presentation_ordinal {
        return;
    }
    if sequence.is_none() {
        let Some(proof) = transition.past_reference_proof else {
            *rejected = true;
            return;
        };
        if proof.reference != previous_identity {
            *rejected = true;
            return;
        }
        match CodecMotionSequence::from_frame(transition) {
            Ok(first) => *sequence = Some(first),
            Err(_) => *rejected = true,
        }
        return;
    }
    if sequence
        .as_mut()
        .expect("sequence existence checked above")
        .push_contiguous(transition)
        .is_err()
    {
        *sequence = None;
        *rejected = true;
    }
}

fn accepted_codec_motion_sequence(
    sequence: Option<CodecMotionSequence>,
    rejected: bool,
) -> Option<CodecMotionProduct> {
    (!rejected).then_some(sequence).flatten().map(Into::into)
}

fn finish_unreached_seek(
    newest: Option<DecodedVideoFrame>,
    reached_eof: bool,
    target_seconds: f64,
    duration_seconds: f64,
    frame_period: f64,
    path: &str,
) -> Result<DecodedVideoFrame, DecodeWorkError> {
    let terminal_selection =
        reached_eof && duration_seconds > 0.0 && target_seconds + frame_period >= duration_seconds;
    if terminal_selection {
        return newest.ok_or_else(|| {
            DecodeWorkError::Failed(format!(
                "video stream in {path} reached EOF without a decodable terminal frame"
            ))
        });
    }
    let reason = if reached_eof {
        "reached EOF before the requested source time"
    } else {
        "exhausted the bounded forward-decode window before the requested source time"
    };
    Err(DecodeWorkError::Failed(format!(
        "could not select {:.6}s in {path}: {reason} (limit {MAX_SEEK_DECODE_FRAMES} frames)",
        target_seconds
    )))
}

fn next_loop_generation(current: u64) -> u64 {
    current.saturating_add(1)
}

fn count_packet_without_frame(count: u32, path: &str) -> Result<u32, String> {
    let next = count.saturating_add(1);
    if next > MAX_PACKETS_WITHOUT_FRAME {
        Err(format!(
            "video stream in {path} yielded no decodable frame after {MAX_PACKETS_WITHOUT_FRAME} packets"
        ))
    } else {
        Ok(next)
    }
}

fn require_decoded_frame_before_loop(frame_count: u64, path: &str) -> Result<(), String> {
    if frame_count == 0 {
        Err(format!(
            "video stream in {path} yielded no decodable frame after a complete pass"
        ))
    } else {
        Ok(())
    }
}

pub(super) fn open_input(
    path: &str,
    cancel: Arc<AtomicBool>,
    operation: &str,
) -> Result<Input, String> {
    let deadline = Instant::now() + MEDIA_IO_TIMEOUT;
    // FFmpeg retains this callback for the Input's lifetime. `opening` turns
    // off the startup deadline after stream probing succeeds; later packet
    // reads remain cancellation-aware without expiring merely because normal
    // playback has lasted longer than five seconds.
    let opening = Arc::new(AtomicBool::new(true));
    let callback_opening = opening.clone();
    let callback_cancel = cancel.clone();
    let result = input_with_interrupt(path, move || {
        should_interrupt_input(
            &callback_cancel,
            &callback_opening,
            Instant::now(),
            deadline,
        )
    });
    opening.store(false, Ordering::Release);
    result.map_err(|error| {
        if cancel.load(Ordering::Acquire) {
            format!("cancelled while trying to {operation} media {path}: {error}")
        } else {
            format!(
                "Cannot {operation} {path} within {} seconds: {error}",
                MEDIA_IO_TIMEOUT.as_secs()
            )
        }
    })
}

fn should_interrupt_input(
    cancel: &AtomicBool,
    opening: &AtomicBool,
    now: Instant,
    deadline: Instant,
) -> bool {
    cancel.load(Ordering::Acquire) || (opening.load(Ordering::Acquire) && now >= deadline)
}

/// Validate dimensions before CPU scaler or GPU texture allocation.
///
/// `adapter_max_dimension` is an additional device-specific edge limit. The
/// aggregate pixel/byte cap always applies even when an adapter advertises a
/// much larger maximum texture edge.
pub fn validate_media_dimensions(
    width: u32,
    height: u32,
    adapter_max_dimension: Option<u32>,
) -> Result<u64, String> {
    validate_safe_dimensions(
        MediaSourceKind::Video,
        width,
        height,
        MediaDeviceLimits {
            max_texture_dimension_2d: adapter_max_dimension,
            max_buffer_size: None,
        },
    )
    .map(|plan| plan.rgba_bytes)
    .map_err(|error| error.to_string())
}

/// Policy-aware dimension planning for callers that have a shared host-local
/// policy. This does not reserve memory; source constructors must retain the
/// [`MediaReservation`] returned by `MediaSafetyPolicy::reserve_source`.
#[allow(dead_code)]
pub fn validate_media_dimensions_with_policy(
    width: u32,
    height: u32,
    source_kind: MediaSourceKind,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<MediaAllocationPlan, String> {
    media_policy
        .plan(source_kind, width, height, device_limits)
        .map_err(|error| error.to_string())
}

/// FFmpeg aligns decoded planes for SIMD, so `stride` is often larger than
/// `width * 4`. wgpu uploads use tightly packed rows; copying the padded plane
/// verbatim would make every row after the first begin at the wrong byte.
fn repack_rgba_plane(
    data: &[u8],
    stride: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let width = usize::try_from(width)
        .map_err(|_| "FFmpeg RGBA width does not fit this platform".to_string())?;
    let height = usize::try_from(height)
        .map_err(|_| "FFmpeg RGBA height does not fit this platform".to_string())?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| "FFmpeg RGBA row byte count overflows".to_string())?;
    let required_len = stride
        .checked_mul(height)
        .ok_or_else(|| "FFmpeg RGBA plane byte count overflows".to_string())?;
    let output_len = row_bytes
        .checked_mul(height)
        .ok_or_else(|| "packed FFmpeg RGBA frame byte count overflows".to_string())?;
    if stride < row_bytes || data.len() < required_len {
        return Err(format!(
            "Invalid FFmpeg RGBA plane: stride={stride}, row_bytes={row_bytes}, height={height}, data_len={}",
            data.len()
        ));
    }

    let mut packed = reserve_packed_rgba(output_len)?;
    if stride == row_bytes {
        packed.extend_from_slice(&data[..output_len]);
        return Ok(packed);
    }

    for row in data.chunks_exact(stride).take(height) {
        packed.extend_from_slice(&row[..row_bytes]);
    }
    Ok(packed)
}

fn reserve_packed_rgba(output_len: usize) -> Result<Vec<u8>, String> {
    let mut packed = Vec::new();
    packed.try_reserve_exact(output_len).map_err(|error| {
        format!("could not reserve {output_len} bytes for packed FFmpeg RGBA frame: {error}")
    })?;
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::{
        can_continue_forward, count_packet_without_frame, enable_export_motion_vectors,
        next_loop_generation, repack_rgba_plane, require_decoded_frame_before_loop,
        reserve_packed_rgba, should_interrupt_input, validate_media_dimensions, StreamPosition,
        VideoDecoder, MAX_PACKETS_WITHOUT_FRAME,
    };
    use crate::video::{CodecFrameIdentity, CodecTimeBase};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    // --- forward-continue law -------------------------------------------
    //
    // Every forward advance used to seek back to the preceding keyframe and
    // re-walk the whole GOP, so a displayed frame cost O(GOP) decodes. These
    // pin the exact conditions under which the walk may instead continue from
    // where the live decoder already is.

    const FRAME_PERIOD: f64 = 1.0 / 30.0;

    fn position_at(source_seconds: f64, pts: i64) -> Option<StreamPosition> {
        Some(StreamPosition {
            source_seconds,
            pts: Some(pts),
        })
    }

    fn accepted(pts: i64) -> Option<CodecFrameIdentity> {
        Some(CodecFrameIdentity {
            source_generation: 1,
            pts,
            presentation_ordinal: 7,
        })
    }

    #[test]
    fn an_unknown_stream_position_always_seeks() {
        assert!(!can_continue_forward(None, 4.0, 0.0, FRAME_PERIOD, None));
    }

    #[test]
    fn a_non_finite_stream_position_always_seeks() {
        for broken in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                !can_continue_forward(position_at(broken, 0), 4.0, 0.0, FRAME_PERIOD, None),
                "{broken} must not authorize a forward walk"
            );
        }
    }

    #[test]
    fn a_keyframe_between_the_position_and_the_target_seeks_to_it() {
        // Position 7.9 s, target 8.5 s, and a keyframe at 8.0 s: seeking lands
        // closer to the target than continuing would, so the walk must seek.
        assert!(!can_continue_forward(
            position_at(7.9, 790),
            8.5,
            8.0,
            FRAME_PERIOD,
            None
        ));
        // The same request with the keyframe behind the position continues.
        assert!(can_continue_forward(
            position_at(7.9, 790),
            8.5,
            0.0,
            FRAME_PERIOD,
            None
        ));
    }

    #[test]
    fn the_refusal_boundary_is_exactly_the_walks_own_reached_test() {
        // The walk returns the first picture satisfying
        // `source_seconds + frame_period * 0.5 >= target_seconds`. Continuing
        // forward is only equivalent while the picture already at the live
        // position fails that test, so the two boundaries must agree exactly.
        let position_seconds = 2.0;
        for step in -4..=4 {
            let target = position_seconds + f64::from(step) * FRAME_PERIOD * 0.25;
            let position_would_satisfy_reached = position_seconds + FRAME_PERIOD * 0.5 >= target;
            let continues = can_continue_forward(
                position_at(position_seconds, 200),
                target,
                0.0,
                FRAME_PERIOD,
                None,
            );
            assert_eq!(
                continues, !position_would_satisfy_reached,
                "target {target} disagrees with the reached test"
            );
        }
    }

    #[test]
    fn a_backward_request_never_continues_forward() {
        assert!(!can_continue_forward(
            position_at(9.0, 900),
            1.0,
            0.0,
            FRAME_PERIOD,
            None
        ));
    }

    #[test]
    fn an_ordinary_one_frame_advance_continues_instead_of_seeking() {
        assert!(can_continue_forward(
            position_at(2.0, 200),
            2.0 + FRAME_PERIOD,
            0.0,
            FRAME_PERIOD,
            None
        ));
    }

    #[test]
    fn composing_motion_requires_the_position_to_be_at_or_after_the_accepted_picture() {
        let target = 2.0 + FRAME_PERIOD;
        // The live decoder sits exactly on the previously accepted picture:
        // the frames it will now decode are precisely the transitions after
        // it, which is what the seek walk would have collected.
        assert!(can_continue_forward(
            position_at(2.0, 200),
            target,
            0.0,
            FRAME_PERIOD,
            accepted(200)
        ));
        // It sits later than the accepted picture, because a decode was
        // superseded. Continuing forward is still admitted — requiring
        // equality here traps the pipeline in a permanent seek-per-frame
        // state — and the omitted transitions make the composed sequence fail
        // its own past-reference proof rather than come out wrong.
        assert!(can_continue_forward(
            position_at(2.0, 200),
            target,
            0.0,
            FRAME_PERIOD,
            accepted(140)
        ));
        // Behind the accepted picture is refused: that is not a position the
        // pipeline should ever be in, so it is not one to optimize.
        assert!(!can_continue_forward(
            position_at(2.0, 200),
            target,
            0.0,
            FRAME_PERIOD,
            accepted(260)
        ));
        // A position with no PTS cannot be proven to be that picture.
        assert!(!can_continue_forward(
            Some(StreamPosition {
                source_seconds: 2.0,
                pts: None,
            }),
            target,
            0.0,
            FRAME_PERIOD,
            accepted(200)
        ));
        // With no accepted identity no sequence is composed, so the condition
        // is vacuous rather than blocking.
        assert!(can_continue_forward(
            Some(StreamPosition {
                source_seconds: 2.0,
                pts: None,
            }),
            target,
            0.0,
            FRAME_PERIOD,
            None
        ));
    }

    #[test]
    fn export_motion_vectors_flag_is_enabled_before_decoder_open() {
        let mut context = ffmpeg_next::codec::context::Context::new();
        enable_export_motion_vectors(&mut context);
        // SAFETY: the test owns the live context and only reads its public ABI
        // flags before it is consumed by a decoder open.
        let flags = unsafe { (*context.as_ptr()).flags2 };
        assert_ne!(flags & ffmpeg_next::ffi::AV_CODEC_FLAG2_EXPORT_MVS, 0);
    }

    #[test]
    fn rgba_plane_padding_is_removed_row_by_row() {
        // Two 3-pixel rows, each padded from 12 to 16 bytes.
        let data: Vec<u8> = (0..32).collect();
        let packed = repack_rgba_plane(&data, 16, 3, 2).unwrap();
        assert_eq!(packed.len(), 24);
        assert_eq!(&packed[..12], &data[..12]);
        assert_eq!(&packed[12..], &data[16..28]);
    }

    #[test]
    fn tightly_packed_plane_is_preserved() {
        let data: Vec<u8> = (0..24).collect();
        assert_eq!(repack_rgba_plane(&data, 12, 3, 2).unwrap(), data);
    }

    #[test]
    fn malformed_plane_is_a_contextual_error_instead_of_a_panic() {
        let error = repack_rgba_plane(&[0; 15], 8, 3, 2).unwrap_err();
        assert!(error.contains("stride=8"));
        assert!(error.contains("row_bytes=12"));
    }

    #[test]
    fn impossible_packed_frame_reservation_is_fallible() {
        let error = reserve_packed_rgba(usize::MAX).unwrap_err();
        assert!(error.contains("could not reserve"));
    }

    #[test]
    fn aggregate_media_cap_rejects_large_but_short_edge_dimensions() {
        assert!(validate_media_dimensions(3840, 2160, Some(16_384)).is_ok());
        assert!(validate_media_dimensions(3840, 2161, Some(16_384)).is_err());
        assert!(validate_media_dimensions(3000, 3000, Some(16_384)).is_err());
        assert!(validate_media_dimensions(8192, 1, Some(8192)).is_ok());
        assert!(validate_media_dimensions(8193, 1, Some(8192)).is_err());
        assert!(validate_media_dimensions(16_385, 1, None).is_err());
    }

    #[test]
    fn open_deadline_is_disabled_after_probe_but_cancel_remains_live() {
        let cancel = AtomicBool::new(false);
        let opening = AtomicBool::new(true);
        let deadline = Instant::now();
        assert!(should_interrupt_input(
            &cancel,
            &opening,
            deadline + Duration::from_secs(1),
            deadline
        ));

        opening.store(false, Ordering::Release);
        assert!(!should_interrupt_input(
            &cancel,
            &opening,
            deadline + Duration::from_secs(60),
            deadline
        ));
        cancel.store(true, Ordering::Release);
        assert!(should_interrupt_input(
            &cancel,
            &opening,
            deadline + Duration::from_secs(60),
            deadline
        ));
    }

    #[test]
    fn zero_frame_stream_cannot_enter_a_reopen_loop() {
        let error = require_decoded_frame_before_loop(0, "empty.mp4").unwrap_err();
        assert!(error.contains("empty.mp4"));
        assert!(error.contains("no decodable frame"));
        assert!(require_decoded_frame_before_loop(1, "loop.mp4").is_ok());
    }

    #[test]
    fn packet_only_stream_has_a_finite_scan_budget() {
        assert_eq!(
            count_packet_without_frame(MAX_PACKETS_WITHOUT_FRAME - 1, "packets.mp4").unwrap(),
            MAX_PACKETS_WITHOUT_FRAME
        );
        let error =
            count_packet_without_frame(MAX_PACKETS_WITHOUT_FRAME, "packets.mp4").unwrap_err();
        assert!(error.contains("packets.mp4"));
        assert!(error.contains("no decodable frame"));
    }

    #[test]
    fn loop_generation_is_monotonic_for_the_decoder_lifetime() {
        assert_eq!(next_loop_generation(0), 1);
        assert_eq!(next_loop_generation(41), 42);
        assert_eq!(next_loop_generation(u64::MAX), u64::MAX);
    }

    #[test]
    fn real_video_eof_reopens_advance_the_cumulative_loop_generation() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("loop-72f.mp4");
        assert!(
            fixture.is_file(),
            "missing decode fixture: {}",
            fixture.display()
        );
        let mut decoder = VideoDecoder::open(&fixture.to_string_lossy()).unwrap();
        let expected_bytes = decoder.width as usize * decoder.height as usize * 4;
        let mut observed_generation = decoder.loop_generation();
        let mut generation_edges = Vec::new();

        // The fixture contains 72 frames. The guard is deliberately much
        // larger than two loops but finite, so a broken EOF/reopen path fails
        // instead of turning the test into an unbounded decoder loop.
        for _ in 0..256 {
            let frame = decoder.next_timed_frame_result(55).unwrap();
            assert_eq!(frame.rgba.len(), expected_bytes);
            assert_eq!(frame.metadata.source_generation, 55);
            assert_eq!(
                frame
                    .codec_motion
                    .as_ref()
                    .map(|motion| motion.source_generation),
                Some(55)
            );
            let generation = decoder.loop_generation();
            if generation != observed_generation {
                assert_eq!(generation, observed_generation + 1);
                generation_edges.push(generation);
                observed_generation = generation;
                if generation == 2 {
                    break;
                }
            }
        }

        assert_eq!(generation_edges, [1, 2]);
    }

    #[test]
    fn metadata_probe_reports_dimensions_without_constructing_playback_state() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("loop-72f.mp4");
        let policy = crate::media_safety::MediaSafetyPolicy::safe();
        let plan = VideoDecoder::probe_dimensions_with_media_policy(
            &fixture.to_string_lossy(),
            &policy,
            crate::media_safety::MediaDeviceLimits::none(),
        )
        .unwrap();
        assert_eq!(
            plan.source_kind,
            crate::media_safety::MediaSourceKind::Video
        );
        assert!(plan.width > 0 && plan.height > 0);
        assert_eq!(plan.pixels, u64::from(plan.width) * u64::from(plan.height));
    }

    #[test]
    fn indexed_reverse_selection_uses_seek_and_cache_without_reopen_or_zero_scan() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("loop-72f.mp4");
        let mut decoder = VideoDecoder::open(&fixture.to_string_lossy()).unwrap();
        let indexed = decoder.build_keyframe_index().unwrap();
        assert!(indexed >= 1);
        assert!(indexed <= super::super::indexed::MAX_KEYFRAME_INDEX_ENTRIES);

        let later = decoder.seek_decode(1.6).unwrap();
        assert!((later.metadata.source_seconds - 1.6).abs() <= 1.0 / 15.0);
        assert!(later.metadata.duration_seconds > later.metadata.source_seconds);
        assert_eq!(
            later
                .codec_motion
                .as_ref()
                .map(|motion| motion.source_generation),
            Some(0)
        );
        let after_forward = decoder.seek_stats();
        assert_eq!(after_forward.index_builds, 1);
        assert_eq!(after_forward.seek_calls, 1);
        assert_eq!(after_forward.reopen_calls, 0);
        assert_eq!(after_forward.scans_from_zero, 0);

        let earlier = decoder.seek_decode_for_generation(1.52, 77).unwrap();
        assert!((earlier.metadata.source_seconds - 1.52).abs() <= 1.0 / 15.0);
        assert_eq!(earlier.metadata.source_generation, 77);
        assert!(earlier.metadata.codec_identity.is_none_or(|identity| {
            identity.source_generation == earlier.metadata.source_generation
                && Some(identity.pts) == earlier.metadata.pts
        }));
        assert!(
            earlier.codec_motion.is_none(),
            "reverse-cache retrieval must not retain stale codec metadata"
        );
        let after_reverse = decoder.seek_stats();
        assert_eq!(after_reverse.seek_calls, after_forward.seek_calls);
        assert_eq!(after_reverse.reverse_cache_hits, 1);
        assert_eq!(after_reverse.reopen_calls, 0);
        assert_eq!(after_reverse.scans_from_zero, 0);

        // The cache already contains the later frame, but a forward request
        // must decode from the index so it can reconstruct current codec
        // proof state instead of returning an RGBA-only cache hit.
        let forward_again = decoder.seek_decode_for_generation(1.6, 78).unwrap();
        assert_eq!(forward_again.metadata.source_generation, 78);
        let after_forward_again = decoder.seek_stats();
        assert_eq!(after_forward_again.seek_calls, after_reverse.seek_calls + 1);
        assert_eq!(
            after_forward_again.reverse_cache_hits,
            after_reverse.reverse_cache_hits
        );
        let (_, cache_bytes) = decoder.reverse_cache_usage();
        assert!(cache_bytes <= super::super::indexed::MAX_REVERSE_CACHE_BYTES);
    }

    #[test]
    fn reverse_cache_direction_has_a_strict_finite_deadband() {
        assert!(super::is_strict_reverse_request(1.0, 1.1, 1.0 / 30.0));
        assert!(!super::is_strict_reverse_request(1.1, 1.0, 1.0 / 30.0));
        assert!(!super::is_strict_reverse_request(1.0, 1.0, 1.0 / 30.0));
        assert!(!super::is_strict_reverse_request(f64::NAN, 1.0, 1.0 / 30.0));
        assert!(!super::is_strict_reverse_request(1.0, 1.1, 0.0));
    }

    #[test]
    fn only_mpeg4_simple_progressive_ip_has_the_previous_anchor_law() {
        use ffmpeg_next::codec::profile::MPEG4;
        use ffmpeg_next::codec::Profile;

        assert!(super::codec_has_previous_anchor_motion(
            ffmpeg_next::codec::Id::MPEG4,
            Profile::MPEG4(MPEG4::Simple),
            false,
        ));
        assert!(!super::codec_has_previous_anchor_motion(
            ffmpeg_next::codec::Id::MPEG4,
            Profile::MPEG4(MPEG4::Simple),
            true,
        ));
        assert!(!super::codec_has_previous_anchor_motion(
            ffmpeg_next::codec::Id::MPEG4,
            Profile::MPEG4(MPEG4::AdvancedSimple),
            false,
        ));
        assert!(!super::codec_has_previous_anchor_motion(
            ffmpeg_next::codec::Id::H264,
            Profile::Unknown,
            false,
        ));
    }

    #[test]
    fn exact_vfr_proof_uses_pts_ticks_and_rejects_hostile_identity_gaps() {
        let reference = CodecFrameIdentity {
            source_generation: 3,
            pts: 100,
            presentation_ordinal: 40,
        };
        let destination = CodecFrameIdentity {
            source_generation: 3,
            pts: 106,
            presentation_ordinal: 41,
        };
        let time_base = CodecTimeBase::new(1, 1_000).unwrap();
        let proof = super::exact_adjacent_reference_proof(
            Some(reference),
            Some(destination),
            Some(time_base),
        )
        .expect("six VFR ticks are an exact adjacent presentation interval");
        assert_eq!(proof.elapsed_ticks, 6);
        assert!((proof.elapsed_seconds().unwrap() - 0.006).abs() < 1.0e-7);
        assert_eq!(
            super::exact_identity_elapsed(Some(reference), Some(destination), time_base),
            Some(0.006)
        );

        for hostile in [
            CodecFrameIdentity {
                presentation_ordinal: 42,
                ..destination
            },
            CodecFrameIdentity {
                source_generation: 4,
                ..destination
            },
            CodecFrameIdentity {
                pts: 100,
                ..destination
            },
        ] {
            assert!(super::exact_adjacent_reference_proof(
                Some(reference),
                Some(hostile),
                Some(time_base),
            )
            .is_none());
        }
    }

    struct TemporaryCodecMotionFixture {
        root: std::path::PathBuf,
        video: std::path::PathBuf,
    }

    impl TemporaryCodecMotionFixture {
        fn create() -> Result<Self, String> {
            Self::create_with_b_frames(2)
        }

        fn create_with_b_frames(b_frames: u8) -> Result<Self, String> {
            let root = std::env::temp_dir().join(format!(
                "collideoscope-codec-motion-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            ));
            std::fs::create_dir(&root).map_err(|error| {
                format!("create temporary codec-motion fixture directory: {error}")
            })?;
            let video = root.join("moving-box.mp4");
            let executable = std::env::var_os("FFMPEG_DIR")
                .map(std::path::PathBuf::from)
                .map(|directory| directory.join("bin").join("ffmpeg.exe"))
                .filter(|candidate| candidate.is_file())
                .unwrap_or_else(|| std::path::PathBuf::from("ffmpeg"));
            let output = std::process::Command::new(executable)
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:size=64x64:rate=12:duration=1,drawbox=x=mod(t*24\\,48):y=24:w=16:h=16:color=white:t=fill",
                    "-c:v",
                    "mpeg4",
                    "-g",
                    "12",
                    "-bf",
                ])
                .arg(b_frames.to_string())
                .args([
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                ])
                .arg(&video)
                .output()
                .map_err(|error| format!("run temporary FFmpeg motion fixture: {error}"))?;
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                let _ = std::fs::remove_dir_all(&root);
                return Err(format!("generate temporary codec-motion fixture: {error}"));
            }
            Ok(Self { root, video })
        }
    }

    /// 120 moving frames at 30 fps inside a single group of pictures. A long
    /// GOP is exactly the shape that makes a seek-per-frame walk expensive.
    struct TemporaryLongGopFixture {
        root: std::path::PathBuf,
        video: std::path::PathBuf,
    }

    impl TemporaryLongGopFixture {
        fn create() -> Result<Self, String> {
            let root = std::env::temp_dir().join(format!(
                "collideoscope-long-gop-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            ));
            std::fs::create_dir(&root)
                .map_err(|error| format!("create temporary long-GOP directory: {error}"))?;
            let video = root.join("long-gop.mp4");
            let executable = std::env::var_os("FFMPEG_DIR")
                .map(std::path::PathBuf::from)
                .map(|directory| directory.join("bin").join("ffmpeg.exe"))
                .filter(|candidate| candidate.is_file())
                .unwrap_or_else(|| std::path::PathBuf::from("ffmpeg"));
            let output = std::process::Command::new(executable)
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x64:rate=30:duration=4",
                    "-c:v",
                    "mpeg4",
                    "-g",
                    "120",
                    "-bf",
                    "0",
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                ])
                .arg(&video)
                .output()
                .map_err(|error| format!("run temporary long-GOP fixture: {error}"))?;
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                let _ = std::fs::remove_dir_all(&root);
                return Err(format!("generate temporary long-GOP fixture: {error}"));
            }
            Ok(Self { root, video })
        }
    }

    impl Drop for TemporaryLongGopFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary long-GOP clip"]
    fn forward_playback_returns_the_seek_walks_pictures_without_re_walking_the_gop() {
        let fixture = TemporaryLongGopFixture::create().unwrap();
        let path = fixture.video.to_string_lossy().to_string();

        // One decoder, played forward the way live playback advances it.
        let mut forward = VideoDecoder::open(&path).unwrap();
        let mut played = Vec::new();
        for frame_index in 0..60u32 {
            let target = f64::from(frame_index) / 30.0;
            played.push(forward.seek_decode(target).unwrap());
        }

        // The baseline is the walk this change replaced: a decoder whose
        // stream position is unknown must seek to the preceding keyframe and
        // decode the whole group of pictures to reach the same target.
        for (frame_index, forward_frame) in played.iter().enumerate() {
            let target = frame_index as f64 / 30.0;
            let mut seeking = VideoDecoder::open(&path).unwrap();
            let seek_frame = seeking.seek_decode(target).unwrap();
            assert_eq!(
                seek_frame.metadata.pts, forward_frame.metadata.pts,
                "frame {frame_index} selected a different picture"
            );
            assert_eq!(
                seek_frame.metadata.source_seconds, forward_frame.metadata.source_seconds,
                "frame {frame_index} reported a different source time"
            );
            assert!(
                seek_frame.rgba == forward_frame.rgba,
                "frame {frame_index} decoded different pixels than the seek walk"
            );
            assert_eq!(
                seeking.seek_stats().seek_calls,
                1,
                "the baseline decoder must actually seek"
            );
        }

        // The whole point: 60 forward advances inside one group of pictures
        // cost one seek, not sixty. Anything above a handful means the fast
        // path stopped engaging and every displayed frame is re-walking.
        assert!(
            forward.seek_stats().seek_calls <= 2,
            "forward playback seeked {} times across one GOP",
            forward.seek_stats().seek_calls
        );
        assert_eq!(forward.seek_stats().reopen_calls, 0);
        assert_eq!(forward.seek_stats().scans_from_zero, 0);
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary long-GOP clip"]
    fn forward_playback_agrees_with_the_seek_walk_all_the_way_through_the_tail() {
        // Playing to the very end is the case where continuing forward and
        // seeking genuinely differ: the forward walk can hit EOF having
        // decoded nothing, while the seek walk re-decodes the whole group of
        // pictures and therefore always holds a terminal frame.
        let fixture = TemporaryLongGopFixture::create().unwrap();
        let path = fixture.video.to_string_lossy().to_string();

        let mut forward = VideoDecoder::open(&path).unwrap();
        for frame_index in 0..150u32 {
            let target = f64::from(frame_index) / 30.0;
            let forward_result = forward.seek_decode(target);

            let mut seeking = VideoDecoder::open(&path).unwrap();
            let seek_result = seeking.seek_decode(target);

            match (&forward_result, &seek_result) {
                (Ok(forward_frame), Ok(seek_frame)) => {
                    assert_eq!(
                        forward_frame.metadata.pts, seek_frame.metadata.pts,
                        "frame {frame_index} selected a different picture"
                    );
                    assert!(
                        forward_frame.rgba == seek_frame.rgba,
                        "frame {frame_index} decoded different pixels"
                    );
                }
                (Err(_), Err(_)) => {}
                (forward_result, seek_result) => panic!(
                    "frame {frame_index} at {target}s disagreed: forward={forward_result:?} seek={seek_result:?}"
                ),
            }
        }
    }

    impl Drop for TemporaryCodecMotionFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary inter-frame clip"]
    fn real_temporary_ffmpeg_fixture_exposes_intra_and_inter_codec_metadata() {
        use crate::video::{AdjacentReferencePolicy, CodecMotionStatus};

        let fixture = TemporaryCodecMotionFixture::create_with_b_frames(0).unwrap();
        let mut decoder = VideoDecoder::open(&fixture.video.to_string_lossy()).unwrap();
        let mut saw_intra = false;
        let mut saw_inter_vectors = false;
        for _ in 0..24 {
            let frame = decoder.next_timed_frame_result(91).unwrap();
            let motion = frame
                .codec_motion
                .expect("video frames carry codec metadata");
            assert_eq!(motion.source_generation, frame.metadata.source_generation);
            assert_eq!(motion.source_generation, 91);
            match motion.status {
                CodecMotionStatus::Intra => {
                    saw_intra = true;
                    assert!(motion.vectors.is_empty());
                }
                CodecMotionStatus::Available => {
                    saw_inter_vectors = true;
                    let proof = motion
                        .past_reference_proof
                        .expect("an available real P frame has typed proof");
                    assert_eq!(
                        proof.policy,
                        AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp
                    );
                    assert_eq!(proof.destination, frame.metadata.codec_identity.unwrap());
                    assert_eq!(
                        proof.destination.presentation_ordinal,
                        proof.reference.presentation_ordinal + 1
                    );
                    let elapsed = proof.elapsed_seconds().unwrap();
                    assert!(motion.vectors.iter().all(|vector| {
                        (vector.seconds_from_reference - elapsed).abs() < 1.0e-6
                    }));
                }
                CodecMotionStatus::FutureOnly
                | CodecMotionStatus::ReferenceUnproven
                | CodecMotionStatus::Unavailable
                | CodecMotionStatus::Rejected(_) => {}
            }
            if saw_intra && saw_inter_vectors {
                break;
            }
        }
        assert!(saw_intra, "generated codec stream exposed no intra frame");
        assert!(
            saw_inter_vectors,
            "generated inter-frame stream exposed no usable past vectors"
        );
        assert!(!fixture.video.starts_with(env!("CARGO_MANIFEST_DIR")));
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary B-frame clip"]
    fn real_b_frame_stream_never_claims_an_unproven_codec_reference_interval() {
        use crate::video::{CodecMotionFrameType, CodecMotionStatus};

        let fixture = TemporaryCodecMotionFixture::create().unwrap();
        let mut decoder = VideoDecoder::open(&fixture.video.to_string_lossy()).unwrap();
        let mut saw_inter = false;
        let mut saw_bidirectional = false;
        for _ in 0..24 {
            let frame = decoder.next_timed_frame_result(92).unwrap();
            let motion = frame
                .codec_motion
                .expect("video frames carry bounded codec metadata");
            if motion.frame_type != CodecMotionFrameType::Intra {
                saw_inter = true;
                saw_bidirectional |= motion.frame_type == CodecMotionFrameType::Bidirectional;
                assert_ne!(motion.status, CodecMotionStatus::Available);
                assert!(motion.vectors.is_empty());
            }
        }
        assert!(
            saw_inter,
            "generated B-frame stream exposed no inter frames"
        );
        assert!(
            saw_bidirectional,
            "FFmpeg fixture did not actually decode a B picture"
        );
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary adjacent-P-frame clip"]
    fn real_temporary_ffmpeg_fixture_composes_every_skipped_adjacent_transition() {
        let fixture = TemporaryCodecMotionFixture::create_with_b_frames(0).unwrap();
        let mut decoder = VideoDecoder::open(&fixture.video.to_string_lossy()).unwrap();
        let generation = 121;
        let previous = decoder
            .seek_decode_for_generation(1.0 / 12.0, generation)
            .unwrap();
        let selected = decoder
            .seek_decode_after_for_generation(
                5.0 / 12.0,
                generation,
                previous.metadata.codec_identity,
            )
            .unwrap();
        let product = selected
            .codec_motion
            .expect("adjacent P frames expose a complete codec-motion chain");
        assert_eq!(product.transition_count(), 4);
        assert_eq!(product.latest().source_generation, generation);
        assert_eq!(product.latest().frame_ordinal, 5);
        let field = product
            .rasterize(crate::motion::MotionLatticeQuality::Live)
            .unwrap()
            .expect("the composed adjacent chain rasterizes");
        assert_eq!(
            field.origin(),
            crate::motion::MotionFieldOrigin::CodecVectors
        );
        assert!(field
            .packed_vectors()
            .iter()
            .copied()
            .map(crate::motion::PackedMotionVector::sample)
            .any(|sample| sample.confidence > 0.0));
    }

    #[test]
    fn skipped_codec_steps_publish_only_a_contiguous_accepted_interval() {
        fn frame(source_seconds: f64, ordinal: u64) -> crate::video::DecodedVideoFrame {
            let destination = crate::video::CodecFrameIdentity {
                source_generation: 7,
                pts: ordinal as i64,
                presentation_ordinal: ordinal,
            };
            let reference = crate::video::CodecFrameIdentity {
                source_generation: 7,
                pts: ordinal.saturating_sub(1) as i64,
                presentation_ordinal: ordinal.saturating_sub(1),
            };
            crate::video::DecodedVideoFrame {
                rgba: vec![0; 64 * 64 * 4],
                metadata: crate::video::FrameMetadata::sanitized(
                    7,
                    Some(ordinal as i64),
                    source_seconds,
                    10.0,
                )
                .with_codec_identity(Some(destination)),
                codec_motion: Some(
                    crate::video::CodecMotionFrame {
                        source_dimensions: [64, 64],
                        frame_delta_seconds: 1.0 / 30.0,
                        source_generation: 7,
                        frame_ordinal: ordinal,
                        algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
                        provenance: crate::video::CodecMotionProvenance::FfmpegExportMvs,
                        frame_type: crate::video::CodecMotionFrameType::Predictive,
                        status: crate::video::CodecMotionStatus::Available,
                        past_reference_proof: Some(crate::video::CodecPastReferenceProof {
                            policy: crate::video::AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
                            reference,
                            destination,
                            elapsed_ticks: 1,
                            time_base: crate::video::CodecTimeBase::new(1, 30).unwrap(),
                        }),
                        vectors: vec![crate::motion::CodecMotionVector {
                            destination: [32, 32],
                            block: [64, 64],
                            motion: [-1, 0],
                            motion_scale: 1,
                            seconds_from_reference: 1.0 / 30.0,
                            reference: crate::motion::CodecReferenceDirection::Past,
                            visibility: 1.0,
                        }],
                    }
                    .into(),
                ),
            }
        }

        let previous = crate::video::CodecFrameIdentity {
            source_generation: 7,
            pts: 10,
            presentation_ordinal: 10,
        };
        let mut sequence = None;
        let mut rejected = false;
        for ordinal in 11..=14 {
            super::append_codec_motion_after(
                &mut sequence,
                &mut rejected,
                &frame(ordinal as f64 / 30.0, ordinal),
                Some(previous),
            );
        }
        let product = super::accepted_codec_motion_sequence(sequence, rejected).unwrap();
        assert_eq!(product.transition_count(), 4);
        assert_eq!(product.frame_ordinal, 14);

        let mut missing_prefix = None;
        let mut missing_prefix_rejected = false;
        super::append_codec_motion_after(
            &mut missing_prefix,
            &mut missing_prefix_rejected,
            &frame(14.0 / 30.0, 14),
            Some(previous),
        );
        assert!(
            super::accepted_codec_motion_sequence(missing_prefix, missing_prefix_rejected)
                .is_none()
        );

        let mut source_cut = None;
        let mut source_cut_rejected = false;
        super::append_codec_motion_after(
            &mut source_cut,
            &mut source_cut_rejected,
            &frame(11.0 / 30.0, 11),
            None,
        );
        assert!(super::accepted_codec_motion_sequence(source_cut, source_cut_rejected).is_none());
    }

    #[test]
    fn bounded_seek_exhaustion_never_returns_an_unreached_frame() {
        let wrong = crate::video::DecodedVideoFrame {
            rgba: vec![1, 2, 3, 4],
            metadata: crate::video::FrameMetadata::sanitized(1, Some(0), 0.0, 1000.0),
            codec_motion: None,
        };
        let error = super::finish_unreached_seek(
            Some(wrong),
            false,
            900.0,
            1000.0,
            1.0 / 30.0,
            "sparse.mp4",
        )
        .unwrap_err();
        assert!(error.to_string().contains("bounded forward-decode window"));

        let terminal = crate::video::DecodedVideoFrame {
            rgba: vec![9],
            metadata: crate::video::FrameMetadata::sanitized(1, Some(99), 9.9, 10.0),
            codec_motion: None,
        };
        assert_eq!(
            super::finish_unreached_seek(
                Some(terminal),
                true,
                10.0,
                10.0,
                1.0 / 30.0,
                "terminal.mp4",
            )
            .unwrap()
            .rgba,
            vec![9]
        );
    }
}
