use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::effects::EffectUniforms;
use crate::spout_in::{SpoutFrame, SpoutIn, SpoutStatus};
use crate::video::threaded::{DecoderHealth, MAX_ADVANCE_FRAMES};
use crate::video::ThreadedDecoder;

/// Bounds decoder/channel work on any one render tick while retaining excess
/// transport debt for subsequent ticks.
pub const MAX_DECODE_FRAMES_PER_TICK: u32 = MAX_ADVANCE_FRAMES;
pub const SPOUT_SOURCE_PREFIX: &str = "spout://";

static NEXT_LAYER_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_layer_id() -> u64 {
    NEXT_LAYER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("live layer identity space exhausted")
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
    Difference,
}

impl BlendMode {
    pub fn as_u32(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Screen => 1,
            BlendMode::Multiply => 2,
            BlendMode::Difference => 3,
        }
    }

    /// Stable lowercase value used by patches and the web protocol.
    /// Keep this separate from the human-facing title-case label so a
    /// snapshot always matches the HTML `<option value>` exactly.
    pub fn key(self) -> &'static str {
        match self {
            BlendMode::Normal => "normal",
            BlendMode::Screen => "screen",
            BlendMode::Multiply => "multiply",
            BlendMode::Difference => "difference",
        }
    }
}

pub enum LayerSource {
    Video(ThreadedDecoder),
    Spout(SpoutIn),
}

impl LayerSource {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Video(_) => "video",
            Self::Spout(_) => "spout",
        }
    }
}

pub struct Layer {
    /// Process-lifetime identity for this live instance. Moving/reordering a
    /// layer retains the ID; constructing a replacement (including patch
    /// load) allocates a new one. It is intentionally absent from patches.
    layer_id: u64,
    /// Stable source identity used when persisting/reopening this layer. This
    /// is the canonical path when canonicalization succeeds, otherwise it is
    /// the caller-provided path verbatim.
    pub source_path: String,
    pub filename: String,
    pub source: LayerSource,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub paused: bool,
    pub visible: bool,
    pub effects: EffectUniforms,
    pub width: u32,
    pub height: u32,
    /// Render-side source error (for example a live frame exceeding GPU
    /// limits). Spout worker errors are merged into `spout_status()`.
    source_error: String,
    // Transport
    pub speed: f32, // 0.25..4.0 playback multiplier (1.0 = normal)
    pub fps: f32,   // target decode FPS (e.g. 30.0)
    /// Legacy public timestamp retained for source compatibility. Transport
    /// cadence is governed by `frame_pacer`; callers no longer need to reset
    /// this after decoding a frame.
    pub last_decode: Instant,
    frame_pacer: FramePacer,
}

/// Accumulates fractional media frames instead of resetting a wall clock each
/// time the renderer happens to consume one. That preserves source cadence
/// across uneven redraw intervals: a delayed redraw produces multiple due
/// frames that can be drained by the caller rather than slowing playback.
#[derive(Debug)]
struct FramePacer {
    last_tick: Instant,
    fractional_frames: f64,
    due_frames: u32,
}

impl FramePacer {
    fn new(now: Instant) -> Self {
        Self {
            last_tick: now,
            fractional_frames: 0.0,
            due_frames: 0,
        }
    }

    fn advance(&mut self, now: Instant, fps: f32, speed: f32) {
        let elapsed = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;

        let fps = if fps.is_finite() && fps > 0.0 {
            fps as f64
        } else {
            30.0
        };
        let speed = if speed.is_finite() {
            speed.max(0.01) as f64
        } else {
            1.0
        };

        self.fractional_frames += elapsed.as_secs_f64() * fps * speed;
        let whole_frames = self.fractional_frames.floor();
        if whole_frames >= 1.0 {
            let newly_due = whole_frames.min(u32::MAX as f64) as u32;
            self.fractional_frames -= newly_due as f64;
            self.due_frames = self.due_frames.saturating_add(newly_due);
        }
    }

    fn reset(&mut self, now: Instant) {
        self.last_tick = now;
        self.fractional_frames = 0.0;
        self.due_frames = 0;
    }
}

impl Layer {
    pub fn new(path: &str, device: &wgpu::Device) -> Result<Self, String> {
        let max_dimension = device.limits().max_texture_dimension_2d;
        let decoder = ThreadedDecoder::open_with_texture_limit(path, max_dimension)?;
        let width = decoder.width;
        let height = decoder.height;
        let fps = decoder.fps;
        validate_source_texture_dimensions(width, height, max_dimension, "video")?;
        let (texture, texture_view) = create_layer_texture(device, width, height, "Layer Texture");

        // Preserve a stable path independently from the short display label.
        // Canonicalization makes relative drag/drop and file-dialog paths
        // deterministic for later patch capture and export.
        let source_path = std::fs::canonicalize(path)
            .unwrap_or_else(|_| std::path::PathBuf::from(path))
            .to_string_lossy()
            .into_owned();

        // Extract just the filename from the path.
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        let effects = EffectUniforms {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };

        let now = Instant::now();
        Ok(Self {
            layer_id: allocate_layer_id(),
            source_path,
            filename,
            source: LayerSource::Video(decoder),
            texture,
            texture_view,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            effects,
            width,
            height,
            source_error: String::new(),
            speed: 1.0,
            fps,
            last_decode: now,
            frame_pacer: FramePacer::new(now),
        })
    }

    /// Create a live Spout receiver layer. The texture is initialized to
    /// transparent black so a missing/warming sender never exposes
    /// uninitialized GPU memory or masks lower layers. `SpoutIn` sanitizes the
    /// requested sender name before this stable `spout://` identity is persisted.
    pub fn new_spout(
        sender_name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, String> {
        let mut receiver = SpoutIn::new(sender_name);
        let sanitized_name = receiver.status().sender_name;
        if sanitized_name.is_empty() {
            return Err("Spout sender name is empty after sanitization".to_string());
        }

        let width = 1;
        let height = 1;
        let (texture, texture_view) =
            create_layer_texture(device, width, height, "Spout Layer Texture");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        receiver.start();

        let effects = EffectUniforms {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };
        let now = Instant::now();
        Ok(Self {
            layer_id: allocate_layer_id(),
            source_path: format!("{SPOUT_SOURCE_PREFIX}{sanitized_name}"),
            filename: format!("Spout: {sanitized_name}"),
            source: LayerSource::Spout(receiver),
            texture,
            texture_view,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            effects,
            width,
            height,
            source_error: String::new(),
            speed: 1.0,
            fps: 30.0,
            last_decode: now,
            frame_pacer: FramePacer::new(now),
        })
    }

    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }

    /// Immutable identity of this live layer instance.
    pub fn layer_id(&self) -> u64 {
        self.layer_id
    }

    pub fn is_video(&self) -> bool {
        matches!(self.source, LayerSource::Video(_))
    }

    pub fn progress(&self) -> f32 {
        match &self.source {
            LayerSource::Video(decoder) => decoder.progress(),
            LayerSource::Spout(_) => 0.0,
        }
    }

    /// Take the newest decoded video frame, including the initial seed that is
    /// available before any transport request. Call this before a paused-layer
    /// early return so paused media displays a defined first frame.
    pub fn take_ready_video_frame(&mut self) -> Result<Option<Vec<u8>>, String> {
        match &mut self.source {
            LayerSource::Video(decoder) => decoder.try_next_frame_result(),
            LayerSource::Spout(_) => Ok(None),
        }
    }

    /// Queue a bounded video advancement request. The returned count is the
    /// number accepted by the worker (zero when its single pending slot is
    /// occupied); only accepted pacing debt is retired.
    pub fn request_video_frames(&mut self, count: u32) -> Result<u32, String> {
        let accepted = match &mut self.source {
            LayerSource::Video(decoder) => decoder.request_frames(count)?,
            LayerSource::Spout(_) => 0,
        };
        self.mark_frames_consumed(accepted);
        Ok(accepted)
    }

    /// Advance the transport clock and queue as much due work as the bounded
    /// decoder accepts this tick.
    pub fn request_due_video_frames(&mut self, speed: f32) -> Result<u32, String> {
        let due = self.frames_due(speed);
        self.request_video_frames(due)
    }

    /// Stable video decoder health for state snapshots.
    pub fn video_health(&self) -> Option<DecoderHealth> {
        match &self.source {
            LayerSource::Video(decoder) => Some(decoder.health()),
            LayerSource::Spout(_) => None,
        }
    }

    pub fn spout_status(&self) -> Option<SpoutStatus> {
        match &self.source {
            LayerSource::Spout(receiver) => {
                let mut status = receiver.status();
                if !self.source_error.is_empty() {
                    status.error.clone_from(&self.source_error);
                }
                Some(status)
            }
            LayerSource::Video(_) => None,
        }
    }

    /// Take at most the newest live frame. The receiver's one-frame slot
    /// discards intermediate images, so a slow render tick never builds lag.
    pub fn try_spout_frame(&self) -> Option<SpoutFrame> {
        match &self.source {
            LayerSource::Spout(receiver) => receiver.try_recv(),
            LayerSource::Video(_) => None,
        }
    }

    /// Resize the GPU source texture atomically when a Spout sender changes
    /// dimensions, then upload the complete RGBA frame.
    pub fn upload_spout_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: SpoutFrame,
    ) -> Result<(), String> {
        if !matches!(self.source, LayerSource::Spout(_)) {
            return Err("cannot upload a Spout frame to a video layer".to_string());
        }
        let expected_len = match (frame.width as usize)
            .checked_mul(frame.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
        {
            Some(len) => len,
            None => {
                let error = format!(
                    "Spout frame dimensions overflow: {}x{}",
                    frame.width, frame.height
                );
                self.source_error.clone_from(&error);
                return Err(error);
            }
        };
        if frame.width == 0 || frame.height == 0 || frame.pixels.len() != expected_len {
            let error = format!(
                "invalid Spout frame {}x{}: expected {expected_len} RGBA bytes, got {}",
                frame.width,
                frame.height,
                frame.pixels.len()
            );
            self.source_error.clone_from(&error);
            return Err(error);
        }
        let max_dimension = device.limits().max_texture_dimension_2d;
        if frame.width > max_dimension || frame.height > max_dimension {
            let error = format!(
                "Spout frame {}x{} exceeds this GPU's {max_dimension}px 2D texture limit",
                frame.width, frame.height
            );
            self.source_error.clone_from(&error);
            return Err(error);
        }

        if frame.width != self.width || frame.height != self.height {
            let (texture, texture_view) =
                create_layer_texture(device, frame.width, frame.height, "Spout Layer Texture");
            self.texture = texture;
            self.texture_view = texture_view;
            self.width = frame.width;
            self.height = frame.height;
            self.effects.resolution = [frame.width as f32, frame.height as f32];
        }
        self.upload_frame(queue, &frame.pixels);
        self.source_error.clear();
        Ok(())
    }

    /// Number of media frames currently due at the given (possibly modulated)
    /// speed. Debt is retired only when the request-driven worker accepts it,
    /// so a busy decoder does not lose transport time. The returned work is
    /// capped per tick; excess debt remains queued.
    pub fn frames_due(&mut self, speed: f32) -> u32 {
        self.frame_pacer.advance(Instant::now(), self.fps, speed);
        self.frame_pacer.due_frames.min(MAX_DECODE_FRAMES_PER_TICK)
    }

    /// Record that `count` due frames were drained (typically while keeping
    /// only the newest decoded image for upload).
    pub fn mark_frames_consumed(&mut self, count: u32) {
        self.frame_pacer.due_frames = self.frame_pacer.due_frames.saturating_sub(count);
        self.last_decode = Instant::now();
    }

    /// Clear pending transport debt and restart timing from now. Call this on
    /// pause/resume transitions so intentional pauses are not interpreted as
    /// renderer stalls that need to be caught up.
    pub fn reset_transport_timing(&mut self) {
        let now = Instant::now();
        self.frame_pacer.reset(now);
        self.last_decode = now;
    }

    pub fn upload_frame(&self, queue: &wgpu::Queue, rgba_data: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn create_layer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, texture_view)
}

pub(crate) fn validate_source_texture_dimensions(
    width: u32,
    height: u32,
    max_dimension: u32,
    source_kind: &str,
) -> Result<(), String> {
    crate::video::decoder::validate_media_dimensions(width, height, Some(max_dimension))
        .map(|_| ())
        .map_err(|error| format!("{source_kind} source rejected: {error}"))
}

/// Parse the stable source identity used for persisted Spout receiver layers.
/// An empty suffix is returned as `Some("")` so callers can distinguish an
/// invalid Spout source from an ordinary video path and report it precisely.
pub fn spout_sender_from_source_path(source_path: &str) -> Option<&str> {
    source_path.strip_prefix(SPOUT_SOURCE_PREFIX)
}

/// Valid video file extensions for drag-and-drop.
pub fn is_video_file(path: &std::path::Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_lowercase().as_str(),
            "mp4" | "webm" | "mov" | "avi" | "mkv"
        ),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        allocate_layer_id, spout_sender_from_source_path, validate_source_texture_dimensions,
        BlendMode, FramePacer, MAX_DECODE_FRAMES_PER_TICK,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn frame_pacer_preserves_fractional_time_and_catches_up() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(start);

        pacer.advance(start + Duration::from_millis(20), 25.0, 1.0);
        assert_eq!(pacer.due_frames, 0);
        assert!((pacer.fractional_frames - 0.5).abs() < 1e-6);

        pacer.advance(start + Duration::from_millis(100), 25.0, 1.0);
        assert_eq!(pacer.due_frames, 2);
        assert!((pacer.fractional_frames - 0.5).abs() < 1e-6);
    }

    #[test]
    fn frame_pacer_reset_discards_pause_debt() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(start);
        pacer.advance(start + Duration::from_secs(2), 30.0, 1.0);
        assert_eq!(pacer.due_frames, 60);
        assert_eq!(pacer.due_frames.min(MAX_DECODE_FRAMES_PER_TICK), 8);

        let resumed = start + Duration::from_secs(5);
        pacer.reset(resumed);
        pacer.advance(resumed + Duration::from_millis(16), 30.0, 1.0);
        assert_eq!(pacer.due_frames, 0);
    }

    #[test]
    fn spout_source_identity_is_unambiguous_and_round_trippable() {
        assert_eq!(
            spout_sender_from_source_path("spout://Resolume Composition"),
            Some("Resolume Composition")
        );
        assert_eq!(spout_sender_from_source_path("spout://"), Some(""));
        assert_eq!(spout_sender_from_source_path("C:\\clips\\spout.mp4"), None);
    }

    #[test]
    fn blend_mode_protocol_keys_match_web_option_values() {
        assert_eq!(BlendMode::Normal.key(), "normal");
        assert_eq!(BlendMode::Screen.key(), "screen");
        assert_eq!(BlendMode::Multiply.key(), "multiply");
        assert_eq!(BlendMode::Difference.key(), "difference");
    }

    #[test]
    fn live_layer_ids_are_nonzero_and_unique() {
        let first = allocate_layer_id();
        let second = allocate_layer_id();
        assert_ne!(first, 0);
        assert_ne!(second, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn source_texture_dimensions_reject_zero_and_gpu_oversize() {
        assert!(validate_source_texture_dimensions(0, 1080, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(1920, 0, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(8193, 1080, 8192, "video").is_err());
        assert!(validate_source_texture_dimensions(1920, 1080, 8192, "video").is_ok());
        assert!(validate_source_texture_dimensions(3840, 2160, 16_384, "video").is_ok());
        assert!(validate_source_texture_dimensions(3840, 2161, 16_384, "video").is_err());
        assert!(validate_source_texture_dimensions(3000, 3000, 16_384, "video").is_err());
    }
}
