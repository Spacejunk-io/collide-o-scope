//! Offline high-quality patch renderer.
//!
//! Renders a patch (layer configs + master effects + NTSC) to an MP4 file
//! at configurable resolution and duration using a headless wgpu device
//! and piping raw RGBA frames to ffmpeg.

use std::io::Write as IoWrite;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::effects::EffectUniforms;
use crate::layers::BlendMode;
use crate::ntsc::NtscState;
use crate::patch::PatchState;
use crate::video::VideoDecoder;

/// Configuration for an offline render job.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration_secs: f32,
    pub output_path: String,
}

/// Shared state for progress/cancellation between the render thread and the UI.
pub struct ExportProgress {
    /// 0..10000 representing 0.0%..100.0%
    pub progress: AtomicU32,
    /// Set to true to request cancellation.
    pub cancel: AtomicBool,
    /// Set to true when the job is complete (success or failure).
    pub done: AtomicBool,
    /// Error message if the job failed (empty = success).
    pub error: std::sync::Mutex<String>,
}

impl ExportProgress {
    pub fn new() -> Self {
        Self {
            progress: AtomicU32::new(0),
            cancel: AtomicBool::new(false),
            done: AtomicBool::new(false),
            error: std::sync::Mutex::new(String::new()),
        }
    }

    pub fn progress_f32(&self) -> f32 {
        self.progress.load(Ordering::Relaxed) as f32 / 10000.0
    }
}

/// Handle to a running export job.
pub struct ExportJob {
    pub progress: Arc<ExportProgress>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ExportJob {
    /// Start an export job on a background thread.
    pub fn start(patch: PatchState, config: ExportConfig, library_folder: &str) -> Self {
        let progress = Arc::new(ExportProgress::new());
        let prog = progress.clone();
        let lib_folder = library_folder.to_string();

        let thread = std::thread::spawn(move || {
            if let Err(e) = run_export(&patch, &config, &prog, &lib_folder) {
                *prog.error.lock().unwrap() = e;
            }
            prog.done.store(true, Ordering::Relaxed);
        });

        Self {
            progress,
            thread: Some(thread),
        }
    }

    /// Check if the job is done.
    pub fn is_done(&self) -> bool {
        self.progress.done.load(Ordering::Relaxed)
    }

    /// Cancel the export and wait for the thread to finish.
    pub fn cancel(&self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }
}

/// Uniforms for the composite shader (must match renderer/state.rs).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

/// Internal layer state for offline rendering.
struct ExportLayer {
    decoder: VideoDecoder,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    effects: EffectUniforms,
    opacity: f32,
    blend_mode: BlendMode,
    speed: f32,
    visible: bool,
    paused: bool,
    width: u32,
    height: u32,
}

fn run_export(
    patch: &PatchState,
    config: &ExportConfig,
    progress: &Arc<ExportProgress>,
    library_folder: &str,
) -> Result<(), String> {
    // --- Create headless wgpu device ---
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("No GPU adapter found: {e}"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Export Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        },
    ))
    .map_err(|e| format!("Failed to create export device: {e}"))?;

    let w = config.width;
    let h = config.height;

    // --- Build pipelines (same as renderer/state.rs) ---
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let effects_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let effects_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fullscreen.wgsl").into()),
    });

    let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Effects Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/effects.wgsl").into()),
    });

    let effects_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export Effects PL"),
        bind_group_layouts: &[
            Some(&effects_bind_group_layout),
            Some(&effects_uniform_layout),
        ],
        immediate_size: 0,
    });

    let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Effects Pipeline"),
        layout: Some(&effects_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &effects_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Composite pipeline
    let composite_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let composite_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let composite_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Composite Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
    });

    let composite_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Export Composite PL"),
            bind_group_layouts: &[
                Some(&composite_bind_group_layout),
                Some(&composite_uniform_layout),
            ],
            immediate_size: 0,
        });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Composite Pipeline"),
        layout: Some(&composite_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // --- Composite textures (same 3-texture scheme as live renderer) ---
    let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST;

    let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Export Composite {i}")),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        })
    });

    let composite_views: [wgpu::TextureView; 3] =
        std::array::from_fn(|i| composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default()));

    // --- Open video decoders for each layer ---
    let mut layers: Vec<ExportLayer> = Vec::new();
    for layer_cfg in &patch.layers {
        let path = format!("{}/{}", library_folder, layer_cfg.filename);
        let decoder = match VideoDecoder::open(&path) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("Export: skipping layer '{}': {e}", layer_cfg.filename);
                continue;
            }
        };

        let lw = decoder.width;
        let lh = decoder.height;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Export Layer Tex"),
            size: wgpu::Extent3d {
                width: lw,
                height: lh,
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

        let mut effects = EffectUniforms::default();
        effects.resolution = [lw as f32, lh as f32];
        layer_cfg.effects.apply_to_uniforms(&mut effects);

        let blend_mode = match layer_cfg.blend_mode.as_str() {
            "screen" => BlendMode::Screen,
            "multiply" => BlendMode::Multiply,
            "difference" => BlendMode::Difference,
            _ => BlendMode::Normal,
        };

        layers.push(ExportLayer {
            decoder,
            texture,
            texture_view,
            effects,
            opacity: layer_cfg.opacity,
            blend_mode,
            speed: layer_cfg.speed,
            visible: layer_cfg.visible,
            paused: layer_cfg.paused,
            width: lw,
            height: lh,
        });
    }

    if layers.is_empty() {
        return Err("No layers could be opened for export".to_string());
    }

    // --- Master effects ---
    let mut master_effects = EffectUniforms::default();
    master_effects.resolution = [w as f32, h as f32];
    patch.master.apply_to_uniforms(&mut master_effects);

    // --- NTSC state ---
    let mut ntsc_state = NtscState::new();
    if let Some(ref ntsc_cfg) = patch.ntsc {
        ntsc_state.params = ntsc_cfg.to_params();
    }
    let base_ntsc = ntsc_state.params.clone();

    // --- Modulation matrix (deterministic: beat derived from frame index) ---
    // Audio and MIDI sources read 0 offline; LFO motion renders identically
    // for the same patch every time.
    let mut mod_matrix = crate::modulation::ModMatrix::new();
    if let Some(ref mod_cfg) = patch.modulation {
        mod_cfg.apply_to_matrix(&mut mod_matrix);
    }

    // --- Temporal effects (feedback/slit-scan), same pass as live ---
    let base_temporal = patch
        .temporal
        .as_ref()
        .map(|t| t.to_params())
        .unwrap_or_default();
    let (temporal_pipeline, temporal_bgl, temporal_ubl) =
        crate::renderer::state::build_temporal_pipeline(&device);
    let (history_texture, history_view) =
        crate::renderer::state::build_history_texture(&device, w, h);
    let mut history_write: usize = 0;

    // --- Readback staging buffer ---
    let bytes_per_row = (w * 4 + 255) & !255;
    let buffer_size = (bytes_per_row * h) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // --- Spawn ffmpeg ---
    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgba",
            "-video_size", &format!("{w}x{h}"),
            "-framerate", &config.fps.to_string(),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            &config.output_path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {e}"))?;

    let mut ffmpeg_stdin = ffmpeg.stdin.take().unwrap();

    // --- Frame loop ---
    let total_frames = (config.fps as f32 * config.duration_secs) as u64;
    let frame_interval = 1.0 / config.fps as f32;

    // Track per-layer frame timing (accumulator-based)
    let mut layer_accumulators: Vec<f32> = vec![0.0; layers.len()];

    for frame_num in 0..total_frames {
        if progress.cancel.load(Ordering::Relaxed) {
            break;
        }

        // Update time uniform for effects (breathe, grain seed, etc.)
        let time = frame_num as f32 * frame_interval;
        master_effects.time = time;

        // Sample the modulation matrix at this frame's beat position and
        // derive the modulated params for this frame (bases untouched).
        let beat = time as f64 * (mod_matrix.clock.bpm as f64 / 60.0);
        mod_matrix.update_at_beat(beat);
        let (mod_master, mod_ntsc, mod_temporal) =
            mod_matrix.modulate(&master_effects, &base_ntsc, &base_temporal);

        // Per-layer modulated values for this frame (bases untouched).
        let layer_mods: Vec<(EffectUniforms, f32)> = layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let lm = mod_matrix.modulate_layer(
                    i,
                    l.opacity,
                    l.speed,
                    l.effects.key_threshold,
                );
                let mut fx = l.effects;
                fx.time = time;
                fx.key_threshold = lm.key_threshold;
                (fx, lm.opacity)
            })
            .collect();

        // Advance decoders based on each layer's (modulated) speed/fps
        for (i, layer) in layers.iter_mut().enumerate() {
            if layer.paused || !layer.visible {
                continue;
            }
            let speed = mod_matrix
                .modulate_layer(i, layer.opacity, layer.speed, layer.effects.key_threshold)
                .speed;
            layer_accumulators[i] += frame_interval * speed;
            let layer_interval = 1.0 / 30.0; // layers decode at 30fps base
            while layer_accumulators[i] >= layer_interval {
                layer_accumulators[i] -= layer_interval;
                if let Some(rgba) = layer.decoder.next_frame() {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &layer.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &rgba,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * layer.width),
                            rows_per_image: Some(layer.height),
                        },
                        wgpu::Extent3d {
                            width: layer.width,
                            height: layer.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            // Update layer effects time
            layer.effects.time = time;
        }

        // --- GPU render ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Frame Encoder"),
        });

        // Render layers composited
        render_layers_export(
            &device,
            &mut encoder,
            &layers,
            &layer_mods,
            &composite_textures,
            &composite_views,
            &effects_pipeline,
            &effects_bind_group_layout,
            &effects_uniform_layout,
            &composite_pipeline,
            &composite_bind_group_layout,
            &composite_uniform_layout,
            &sampler,
            w,
            h,
        );

        // Master effects
        render_master_effects_export(
            &device,
            &mut encoder,
            &mod_master,
            &composite_textures,
            &composite_views,
            &effects_pipeline,
            &effects_bind_group_layout,
            &effects_uniform_layout,
            &sampler,
            w,
            h,
        );

        // Temporal effects + history recording (identical pass to live)
        history_write = crate::renderer::state::encode_temporal(
            &device,
            &mut encoder,
            &mod_temporal,
            &temporal_pipeline,
            &temporal_bgl,
            &temporal_ubl,
            &sampler,
            &composite_textures,
            &composite_views,
            &history_texture,
            &history_view,
            history_write,
            w,
            h,
        );

        // Submit GPU work
        queue.submit(std::iter::once(encoder.finish()));

        // --- NTSC at full resolution (modulated params for this frame) ---
        let mut pixels = readback_pixels(&device, &queue, &composite_textures[0], &staging, w, h, bytes_per_row);
        ntsc_state.params = mod_ntsc;
        ntsc_state.apply_full_res(&mut pixels, w, h);

        // Write to ffmpeg
        if ffmpeg_stdin.write_all(&pixels).is_err() {
            break;
        }

        // Update progress
        progress.progress.store(
            ((frame_num + 1) * 10000 / total_frames) as u32,
            Ordering::Relaxed,
        );
    }

    // Close ffmpeg
    drop(ffmpeg_stdin);
    let output = ffmpeg.wait_with_output().map_err(|e| format!("ffmpeg wait: {e}"))?;
    if !output.status.success() && !progress.cancel.load(Ordering::Relaxed) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg failed: {}", stderr.chars().take(200).collect::<String>()));
    }

    Ok(())
}

/// Readback composite_textures[0] to CPU as RGBA bytes (no row padding).
fn readback_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    staging: &wgpu::Buffer,
    w: u32,
    h: u32,
    bytes_per_row: u32,
) -> Vec<u8> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let row_bytes = (w * 4) as usize;
    let padded_row = bytes_per_row as usize;
    let mut pixels = Vec::with_capacity(row_bytes * h as usize);
    for row in 0..h as usize {
        let start = row * padded_row;
        pixels.extend_from_slice(&data[start..start + row_bytes]);
    }
    drop(data);
    staging.unmap();

    pixels
}

/// Render all visible layers composited (mirrors Renderer::render_layers).
fn render_layers_export(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let visible_layers: Vec<(&ExportLayer, &(EffectUniforms, f32))> = layers
        .iter()
        .zip(layer_mods.iter())
        .filter(|(l, _)| l.visible)
        .rev()
        .collect();

    if visible_layers.is_empty() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[0],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        return;
    }

    for (i, (layer, (mod_fx, mod_opacity))) in visible_layers.iter().enumerate() {
        let fx_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Export Layer FX"),
            contents: bytemuck::cast_slice(&[*mod_fx]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if i == 0 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[1],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            let comp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Export Composite Uniforms"),
                contents: bytemuck::cast_slice(&[CompositeUniforms {
                    opacity: *mod_opacity,
                    blend_mode: layer.blend_mode.as_u32(),
                    _pad: [0.0; 2],
                }]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&composite_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: comp_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Export Composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &composite_views[2],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(composite_pipeline);
                pass.set_bind_group(0, &composite_tex_bg, &[]);
                pass.set_bind_group(1, &composite_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[2],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Apply master effects to composite_textures[0] (mirrors Renderer::render_master_effects).
fn render_master_effects_export(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    master_uniforms: &EffectUniforms,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let fx_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Export Master FX"),
        contents: bytemuck::cast_slice(&[*master_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: fx_buffer.as_entire_binding(),
        }],
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Master FX"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[2],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(effects_pipeline);
        pass.set_bind_group(0, &tex_bg, &[]);
        pass.set_bind_group(1, &uniform_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[2],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: output_width,
            height: output_height,
            depth_or_array_layers: 1,
        },
    );
}
