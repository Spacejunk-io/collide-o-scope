use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{context::Input, input_with_interrupt};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Ordinary media is capped at UHD independently of the active GPU's edge
/// limit. A permissive adapter must not turn corrupt codec metadata into a
/// multi-gigabyte scaler or texture allocation.
pub const MAX_MEDIA_PIXELS: u64 = 3_840 * 2_160;
pub const MAX_MEDIA_RGBA_BYTES: u64 = MAX_MEDIA_PIXELS * 4;
pub const MAX_MEDIA_EDGE: u32 = 16_384;
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
    frame_count: u64,
    total_frames: u64,
    cancel: Arc<AtomicBool>,
}

impl VideoDecoder {
    pub fn open(path: &str) -> Result<Self, String> {
        Self::open_with_cancel(path, Arc::new(AtomicBool::new(false)))
    }

    /// Open a decoder with a cooperative cancellation flag shared by the
    /// caller. FFmpeg's interrupt callback observes both cancellation and a
    /// fixed I/O deadline during open and every later packet read/reopen.
    pub fn open_with_cancel(path: &str, cancel: Arc<AtomicBool>) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))?;

        let input_ctx = open_input(path, cancel.clone(), "open")?;

        let stream = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream found")?;
        let stream_index = stream.index();

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder: {e}"))?;

        let width = decoder.width();
        let height = decoder.height();
        validate_media_dimensions(width, height, None)?;
        let avg_fps = f64::from(stream.avg_frame_rate());
        let fps = if avg_fps.is_finite() && avg_fps > 0.0 {
            avg_fps as f32
        } else {
            30.0
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
            frame_count: 0,
            total_frames,
            cancel,
        })
    }

    /// Get the next decoded RGBA frame, preserving contextual decoder/scaler
    /// failures. The stream loops back to the start when the file ends.
    pub fn next_frame_result(&mut self) -> Result<Vec<u8>, String> {
        let mut packets_without_frame = 0u32;
        loop {
            self.check_cancelled()?;
            // Try to receive already-decoded frames first
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                self.frame_count += 1;
                return self.scale_frame(&decoded);
            }

            // Feed more packets to the decoder
            match self.next_video_packet(&mut packets_without_frame)? {
                Some(packet) => {
                    self.decoder.send_packet(&packet).map_err(|e| {
                        format!("Failed to submit packet while decoding {}: {e}", self.path)
                    })?;
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return self.scale_frame(&decoded);
                    }
                }
                None => {
                    // EOF — flush decoder then loop the file
                    self.decoder.send_eof().ok();
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return self.scale_frame(&decoded);
                    }
                    // One zero-frame pass is malformed/unsupported input, not
                    // a looping video. Reopening it forever would wedge layer
                    // startup and offline export cancellation.
                    require_decoded_frame_before_loop(self.frame_count, &self.path)?;
                    // This call already decoded at least one frame before
                    // reaching EOF, so reopen for normal loop playback.
                    self.reopen()?;
                    packets_without_frame = 0;
                }
            }
        }
    }

    /// Returns loop progress as 0.0..1.0
    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.frame_count % self.total_frames) as f32 / self.total_frames as f32
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

    fn next_video_packet(
        &mut self,
        packets_without_frame: &mut u32,
    ) -> Result<Option<ffmpeg::Packet>, String> {
        for (stream, packet) in self.input_ctx.packets() {
            if self.cancel.load(Ordering::Acquire) {
                return Err(format!("video decode cancelled for {}", self.path));
            }
            if stream.index() == self.stream_index {
                *packets_without_frame =
                    count_packet_without_frame(*packets_without_frame, &self.path)?;
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
        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;
        validate_media_dimensions(decoder.width(), decoder.height(), None)?;
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

        Ok(())
    }

    fn check_cancelled(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Acquire) {
            Err(format!("video decode cancelled for {}", self.path))
        } else {
            Ok(())
        }
    }
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
    if width == 0 || height == 0 {
        return Err(format!(
            "media reported invalid dimensions {width}x{height}"
        ));
    }
    if width > MAX_MEDIA_EDGE || height > MAX_MEDIA_EDGE {
        return Err(format!(
            "media dimensions {width}x{height} exceed the {MAX_MEDIA_EDGE}px safety edge limit"
        ));
    }
    if let Some(limit) = adapter_max_dimension {
        if width > limit || height > limit {
            return Err(format!(
                "media dimensions {width}x{height} exceed this GPU's {limit}px 2D texture limit"
            ));
        }
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| format!("media pixel count overflows for {width}x{height}"))?;
    let rgba_bytes = pixels
        .checked_mul(4)
        .ok_or_else(|| format!("media RGBA byte size overflows for {width}x{height}"))?;
    if pixels > MAX_MEDIA_PIXELS || rgba_bytes > MAX_MEDIA_RGBA_BYTES {
        return Err(format!(
            "media dimensions {width}x{height} require {pixels} pixels/{rgba_bytes} RGBA bytes; the safety limit is {MAX_MEDIA_PIXELS} pixels/{MAX_MEDIA_RGBA_BYTES} bytes (3840x2160)"
        ));
    }
    Ok(rgba_bytes)
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
    let row_bytes = width as usize * 4;
    let height = height as usize;
    let required_len = stride.saturating_mul(height);
    if stride < row_bytes || data.len() < required_len {
        return Err(format!(
            "Invalid FFmpeg RGBA plane: stride={stride}, row_bytes={row_bytes}, height={height}, data_len={}",
            data.len()
        ));
    }

    if stride == row_bytes {
        return Ok(data[..row_bytes * height].to_vec());
    }

    let mut packed = Vec::with_capacity(row_bytes * height);
    for row in data.chunks_exact(stride).take(height) {
        packed.extend_from_slice(&row[..row_bytes]);
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::{
        count_packet_without_frame, repack_rgba_plane, require_decoded_frame_before_loop,
        should_interrupt_input, validate_media_dimensions, MAX_PACKETS_WITHOUT_FRAME,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

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
}
