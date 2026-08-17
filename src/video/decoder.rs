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
use crate::video::CodecMotionFrame;

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

        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        enable_export_motion_vectors(&mut codec_ctx);
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder: {e}"))?;

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
        if let Some(frame) = self
            .reverse_cache
            .near_at_or_before(target_seconds, source_generation, frame_period * 1.5)
            .map_err(DecodeWorkError::Failed)?
        {
            self.check_work_current(&mut is_current)?;
            self.seek_stats.reverse_cache_hits =
                self.seek_stats.reverse_cache_hits.saturating_add(1);
            self.adopt_selected_metadata(frame.metadata);
            return Ok(frame);
        }

        let keyframe = self.keyframe_index.preceding(target_seconds);
        self.seek_to_preceding_keyframe(keyframe.pts)
            .map_err(DecodeWorkError::Failed)?;
        self.check_work_current(&mut is_current)?;
        self.frame_count = (keyframe.source_seconds * f64::from(self.fps))
            .floor()
            .max(0.0) as u64;

        let mut newest = None;
        let mut reached_eof = false;
        for _ in 0..MAX_SEEK_DECODE_FRAMES {
            self.check_work_current(&mut is_current)?;
            let Some(frame) =
                self.next_frame_without_loop_interruptible(source_generation, &mut is_current)?
            else {
                reached_eof = true;
                break;
            };
            let reached = frame.metadata.source_seconds + frame_period * 0.5 >= target_seconds;
            if reached {
                return Ok(frame);
            }
            newest = Some(frame);
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
        let metadata = FrameMetadata::sanitized(
            source_generation,
            pts,
            source_seconds,
            self.duration_seconds,
        );
        let rgba = self.scale_frame(frame)?;
        let codec_motion = Some(CodecMotionFrame::from_decoded_frame(
            frame,
            [self.width, self.height],
            1.0 / self.fps,
            source_generation,
            self.frame_count.saturating_sub(1),
        ));
        let decoded = DecodedVideoFrame {
            rgba,
            metadata,
            codec_motion,
        };
        self.adopt_selected_metadata(metadata);
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
        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        enable_export_motion_vectors(&mut codec_ctx);
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;
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

        self.frame_count = 0;
        self.loop_generation = next_loop_generation(self.loop_generation);
        self.last_pts = None;
        self.last_source_seconds = 0.0;
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

fn open_input(path: &str, cancel: Arc<AtomicBool>, operation: &str) -> Result<Input, String> {
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
        count_packet_without_frame, enable_export_motion_vectors, next_loop_generation,
        repack_rgba_plane, require_decoded_frame_before_loop, reserve_packed_rgba,
        should_interrupt_input, validate_media_dimensions, VideoDecoder, MAX_PACKETS_WITHOUT_FRAME,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

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
        assert!(
            earlier.codec_motion.is_none(),
            "reverse-cache retrieval must not retain stale codec metadata"
        );
        let after_reverse = decoder.seek_stats();
        assert_eq!(after_reverse.seek_calls, after_forward.seek_calls);
        assert_eq!(after_reverse.reverse_cache_hits, 1);
        assert_eq!(after_reverse.reopen_calls, 0);
        assert_eq!(after_reverse.scans_from_zero, 0);
        let (_, cache_bytes) = decoder.reverse_cache_usage();
        assert!(cache_bytes <= super::super::indexed::MAX_REVERSE_CACHE_BYTES);
    }

    struct TemporaryCodecMotionFixture {
        root: std::path::PathBuf,
        video: std::path::PathBuf,
    }

    impl TemporaryCodecMotionFixture {
        fn create() -> Result<Self, String> {
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
                    "2",
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

    impl Drop for TemporaryCodecMotionFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates only a temporary inter-frame clip"]
    fn real_temporary_ffmpeg_fixture_exposes_intra_and_inter_codec_metadata() {
        use crate::video::CodecMotionStatus;

        let fixture = TemporaryCodecMotionFixture::create().unwrap();
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
                CodecMotionStatus::Available => saw_inter_vectors = true,
                CodecMotionStatus::FutureOnly
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
