use std::time::Instant;

use crate::effects::EffectUniforms;
use crate::video::ThreadedDecoder;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
    Difference,
}

impl BlendMode {
    pub const ALL: &[BlendMode] = &[
        BlendMode::Normal,
        BlendMode::Screen,
        BlendMode::Multiply,
        BlendMode::Difference,
    ];

    pub fn as_u32(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Screen => 1,
            BlendMode::Multiply => 2,
            BlendMode::Difference => 3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Screen => "Screen",
            BlendMode::Multiply => "Multiply",
            BlendMode::Difference => "Difference",
        }
    }
}

pub struct Layer {
    pub filename: String,
    pub decoder: ThreadedDecoder,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub paused: bool,
    pub visible: bool,
    pub effects: EffectUniforms,
    pub width: u32,
    pub height: u32,
    // Transport
    pub speed: f32,           // 0.25..4.0 playback multiplier (1.0 = normal)
    pub fps: f32,             // target decode FPS (e.g. 30.0)
    pub last_decode: Instant, // tracks per-layer frame timing
}

impl Layer {
    pub fn new(
        path: &str,
        device: &wgpu::Device,
    ) -> Result<Self, String> {
        let decoder = ThreadedDecoder::open(path)?;
        let width = decoder.width;
        let height = decoder.height;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Layer Texture"),
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

        // Extract just the filename from the path
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        let mut effects = EffectUniforms::default();
        effects.resolution = [width as f32, height as f32];

        Ok(Self {
            filename,
            decoder,
            texture,
            texture_view,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            effects,
            width,
            height,
            speed: 1.0,
            fps: 30.0,
            last_decode: Instant::now(),
        })
    }

    /// Returns true if enough time has elapsed for this layer to decode its
    /// next frame at the given (possibly modulated) speed multiplier.
    pub fn ready_for_frame(&self, speed: f32) -> bool {
        let interval = std::time::Duration::from_secs_f32(1.0 / (self.fps * speed.max(0.01)));
        self.last_decode.elapsed() >= interval
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
