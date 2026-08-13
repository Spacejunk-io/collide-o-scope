use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::effects::params::{normalized_slit_direction, TemporalParams, TEMPORAL_REFERENCE_FPS};
use crate::effects::EffectUniforms;
use crate::layers::Layer;

/// Frames of output history kept for temporal effects (0.8s at 30fps).
pub const HISTORY_LEN: u32 = 24;

/// Uniforms for the temporal (feedback/slit-scan) pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TemporalUniforms {
    feedback: f32,
    fb_zoom: f32,
    fb_rotate: f32,
    slitscan: f32,
    history_len: f32,
    write_index: f32,
    valid_history: f32,
    feedback_valid: f32,
    slit_direction: [f32; 2],
    _pad: [f32; 2],
}

/// CPU-side lifetime for temporal GPU memories.
///
/// History snapshots advance at a fixed 30 Hz regardless of render cadence,
/// while `history_valid` and `feedback_valid` prevent the shader from ever
/// touching an unwritten texture. The texture contents themselves do not need
/// an eager clear because invalid layers remain unreachable.
#[derive(Debug, Clone)]
pub(crate) struct TemporalState {
    history_write: usize,
    history_valid: u32,
    history_accumulator: f64,
    feedback_valid: bool,
    initialized: bool,
    total_history_frames: usize,
}

impl Default for TemporalState {
    fn default() -> Self {
        Self {
            history_write: 0,
            history_valid: 0,
            history_accumulator: 0.0,
            feedback_valid: false,
            initialized: false,
            total_history_frames: 0,
        }
    }
}

impl TemporalState {
    /// Reset validity immediately. Stale GPU pixels can remain allocated: the
    /// shader cannot sample them until fresh frames make them valid again.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Number of 30-Hz history snapshots to record for this render step.
    /// The first rendered clean frame primes the ring immediately. Long stalls
    /// are bounded to one complete ring because further copies of the same
    /// current frame cannot add information.
    fn history_writes_for_delta(&mut self, delta_seconds: f32) -> u32 {
        if !self.initialized {
            self.initialized = true;
            return 1;
        }

        let reference_delta = 1.0 / TEMPORAL_REFERENCE_FPS as f64;
        let delta = if delta_seconds.is_finite() {
            delta_seconds.max(0.0) as f64
        } else {
            reference_delta
        };
        self.history_accumulator += delta;

        let elapsed_steps = (self.history_accumulator / reference_delta).floor() as u64;
        if elapsed_steps == 0 {
            return 0;
        }
        self.history_accumulator -= elapsed_steps as f64 * reference_delta;
        elapsed_steps.min(HISTORY_LEN as u64) as u32
    }

    fn record_history_frame(&mut self) {
        if self.history_valid > 0 {
            self.history_write = (self.history_write + 1) % HISTORY_LEN as usize;
        }
        self.history_valid = (self.history_valid + 1).min(HISTORY_LEN);
        self.total_history_frames = self.total_history_frames.saturating_add(1);
    }
}

#[cfg(test)]
mod temporal_state_tests {
    use super::*;

    fn advance(state: &mut TemporalState, delta_seconds: f32) -> u32 {
        let writes = state.history_writes_for_delta(delta_seconds);
        for _ in 0..writes {
            state.record_history_frame();
        }
        writes
    }

    #[test]
    fn history_is_primed_then_advances_at_thirty_hz() {
        let mut state = TemporalState::default();

        assert_eq!(advance(&mut state, 1.0 / 60.0), 1);
        assert_eq!(state.history_valid, 1);
        assert_eq!(advance(&mut state, 1.0 / 60.0), 0);
        assert_eq!(advance(&mut state, 1.0 / 60.0), 1);
        assert_eq!(state.history_valid, 2);
        assert_eq!(state.history_write, 1);
    }

    #[test]
    fn valid_history_never_exceeds_the_ring() {
        let mut state = TemporalState::default();
        advance(&mut state, 0.0);
        for _ in 0..(HISTORY_LEN * 2) {
            advance(&mut state, 1.0 / TEMPORAL_REFERENCE_FPS);
        }

        assert_eq!(state.history_valid, HISTORY_LEN);
        assert!(state.history_write < HISTORY_LEN as usize);
    }

    #[test]
    fn temporal_uniform_layout_matches_three_shader_vec4s() {
        assert_eq!(std::mem::size_of::<TemporalUniforms>(), 48);
    }

    #[test]
    fn reset_revokes_all_temporal_validity() {
        let mut state = TemporalState::default();
        advance(&mut state, 0.0);
        state.feedback_valid = true;

        state.reset();

        assert_eq!(state.history_valid, 0);
        assert!(!state.feedback_valid);
        assert!(!state.initialized);
        assert_eq!(state.total_history_frames, 0);
    }

    #[test]
    fn output_capability_selection_handles_empty_and_fallback_lists() {
        assert_eq!(preferred_surface_format(&[]), None);
        assert_eq!(preferred_present_mode(&[]), None);
        assert_eq!(
            preferred_surface_format(&[wgpu::TextureFormat::Rgba8Unorm]),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
        assert_eq!(
            preferred_present_mode(&[wgpu::PresentMode::Immediate]),
            Some(wgpu::PresentMode::Immediate)
        );
        assert_eq!(
            preferred_present_mode(&[wgpu::PresentMode::Immediate, wgpu::PresentMode::Fifo]),
            Some(wgpu::PresentMode::Fifo)
        );
    }
}

// Readback slot lifecycle for the async NTSC pipeline.
const SLOT_IDLE: u8 = 0;
const SLOT_MAP_PENDING: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_MAP_FAILED: u8 = 3;
const MAX_READBACK_SLOTS: usize = 3;

/// A staging buffer that can have a GPU→CPU copy in flight without
/// blocking the render thread.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    /// Monotonic order in which the GPU copy was submitted.
    sequence: u64,
    /// Application-owned visual generation associated with this copy.
    /// Consumers use it to reject frames captured before a blackout edge.
    epoch: u64,
}

/// A completed asynchronous composite readback.
///
/// `epoch` is opaque to the renderer. The live application advances it at
/// blackout transitions so delayed GPU/CPU work can never reveal an older
/// visual generation.
pub struct ReadbackFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
}

/// Build the temporal pipeline and its layouts. Shared by the live
/// renderer and the offline exporter so both apply identical passes.
pub(crate) fn build_temporal_pipeline(
    device: &wgpu::Device,
) -> (
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Temporal BGL"),
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
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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
            // Last frame's post-temporal output — feedback compounds on this.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Temporal Uniform BGL"),
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

    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Temporal Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/temporal.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Temporal Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout), Some(&uniform_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Temporal Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
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

    (pipeline, bind_group_layout, uniform_layout)
}

/// Create the frame-history texture array and its D2Array view.
pub(crate) fn build_history_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Frame History"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: HISTORY_LEN,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// Single texture holding last frame's post-temporal output, for
/// compounding feedback trails.
pub(crate) fn build_feedback_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Feedback Frame"),
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Encode the temporal pass + history recording. `history_write` is the
/// layer holding the most recent completed frame; returns the new value.
/// Shared by the live renderer and the offline exporter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_temporal_with_dt(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    params: &TemporalParams,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    history_texture: &wgpu::Texture,
    history_view: &wgpu::TextureView,
    feedback_texture: &wgpu::Texture,
    feedback_view: &wgpu::TextureView,
    state: &mut TemporalState,
    delta_seconds: f32,
    width: u32,
    height: u32,
) {
    // Record the CLEAN composite into the ring first — slit-scan must read
    // real past frames, never its own output (which would self-cannibalize
    // into black). new_write becomes the ring's "now".
    let writes = state.history_writes_for_delta(delta_seconds);
    for _ in 0..writes {
        state.record_history_frame();
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: history_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: state.history_write as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    let frame_params = params.for_frame_delta(delta_seconds);
    if frame_params.is_active() {
        let uniforms = TemporalUniforms {
            feedback: frame_params.feedback,
            fb_zoom: frame_params.fb_zoom,
            fb_rotate: frame_params.fb_rotate,
            slitscan: frame_params.slitscan,
            history_len: HISTORY_LEN as f32,
            write_index: state.history_write as f32,
            valid_history: state.history_valid as f32,
            feedback_valid: if state.feedback_valid { 1.0 } else { 0.0 },
            slit_direction: normalized_slit_direction(frame_params.slit_angle, width, height),
            _pad: [0.0; 2],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Temporal Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Temporal Textures BG"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(feedback_view),
                },
            ],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Temporal Uniform BG"),
            layout: uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Temporal Pass"),
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
            pass.set_pipeline(pipeline);
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
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    // Record the finished (post-temporal) frame for next frame's trails.
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: feedback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    state.feedback_valid = true;
}

/// Uniforms for the composite shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // Effects pipeline (per-layer: applies pixelate/rgb_split/color to a single layer)
    effects_pipeline: wgpu::RenderPipeline,
    effects_bind_group_layout: wgpu::BindGroupLayout,
    effects_uniform_layout: wgpu::BindGroupLayout,

    // Composite pipeline (blends overlay onto base)
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_uniform_layout: wgpu::BindGroupLayout,

    // Shared sampler
    sampler: wgpu::Sampler,

    // Three textures for compositing:
    // [0] = accumulated result (base)
    // [1] = current layer after effects (overlay)
    // [2] = composite output (becomes new base)
    pub composite_textures: [wgpu::Texture; 3],
    pub composite_views: [wgpu::TextureView; 3],

    // The view egui displays (always points at the final accumulated result)
    pub output_view: wgpu::TextureView,

    pub output_width: u32,
    pub output_height: u32,

    // Staging buffers for async NTSC readback (created lazily, reused)
    readback_slots: Vec<ReadbackSlot>,
    next_readback_sequence: u64,
    last_harvested_readback_sequence: u64,

    // Temporal effects: ring buffer of past output frames (texture array)
    temporal_pipeline: wgpu::RenderPipeline,
    temporal_bind_group_layout: wgpu::BindGroupLayout,
    temporal_uniform_layout: wgpu::BindGroupLayout,
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,
    feedback_texture: wgpu::Texture,
    feedback_view: wgpu::TextureView,
    /// Fixed-rate history clock and validity for temporal GPU memories.
    temporal_state: TemporalState,

    // Instance + adapter kept for creating additional surfaces (output
    // window). Capabilities must be queried against the SAME adapter the
    // device came from — a freshly requested adapter handle invalidates
    // the device when its capabilities disagree.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    // Dedicated fullscreen output window (projector), when open
    output: Option<OutputTarget>,
}

/// A second window showing the final composite, letterboxed — the face the
/// audience sees while the main window and web panel stay with the performer.
struct OutputTarget {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

fn preferred_surface_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats
        .iter()
        .find(|format| format.is_srgb())
        .copied()
        .or_else(|| formats.first().copied())
}

fn preferred_present_mode(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    if modes.contains(&wgpu::PresentMode::Fifo) {
        Some(wgpu::PresentMode::Fifo)
    } else {
        modes.first().copied()
    }
}

impl Renderer {
    pub fn new(window: Arc<Window>, output_width: u32, output_height: u32) -> Result<Self, String> {
        // Reject corrupt/hostile initial dimensions before constructing any
        // full-frame GPU resource. Adapter limits remain an additional check,
        // never the sole memory-safety boundary.
        crate::video::decoder::validate_media_dimensions(output_width, output_height, None)
            .map_err(|error| format!("invalid renderer output dimensions: {error}"))?;
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|error| format!("failed to create renderer surface: {error}"))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|error| format!("no suitable GPU adapter found: {error}"))?;

        crate::video::decoder::validate_media_dimensions(
            output_width,
            output_height,
            Some(adapter.limits().max_texture_dimension_2d),
        )
        .map_err(|error| format!("unsupported renderer output dimensions: {error}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|error| format!("failed to create renderer device: {error}"))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Effects pipeline (single texture + uniforms → render target) ---
        let effects_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Effects Texture BGL"),
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
                label: Some("Effects Uniform BGL"),
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
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
        });

        let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Effects Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/effects.wgsl").into()),
        });

        let effects_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Effects Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&effects_bind_group_layout),
                    Some(&effects_uniform_layout),
                ],
                immediate_size: 0,
            });

        let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Effects Pipeline"),
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

        // --- Composite pipeline (two textures + uniforms → render target) ---
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite BGL"),
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
                label: Some("Composite Uniform BGL"),
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
            label: Some("Composite Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&composite_bind_group_layout),
                    Some(&composite_uniform_layout),
                ],
                immediate_size: 0,
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
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

        // --- Temporal pipeline + frame-history ring (shared with export) ---
        let (temporal_pipeline, temporal_bind_group_layout, temporal_uniform_layout) =
            build_temporal_pipeline(&device);
        let (history_texture, history_view) =
            build_history_texture(&device, output_width, output_height);
        let (feedback_texture, feedback_view) =
            build_feedback_texture(&device, output_width, output_height);

        // --- Three composite textures ---
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;

        let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Composite {i}")),
                size: wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
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

        let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
            composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
        });

        let output_view =
            composite_textures[0].create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            surface,
            device,
            queue,
            config,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_uniform_layout,
            sampler,
            composite_textures,
            composite_views,
            output_view,
            output_width,
            output_height,
            readback_slots: Vec::new(),
            next_readback_sequence: 1,
            last_harvested_readback_sequence: 0,
            temporal_pipeline,
            temporal_bind_group_layout,
            temporal_uniform_layout,
            history_texture,
            history_view,
            feedback_texture,
            feedback_view,
            temporal_state: TemporalState::default(),
            instance,
            adapter,
            output: None,
        })
    }

    /// Open the fullscreen output window's rendering surface.
    pub fn create_output(&mut self, window: Arc<Window>) -> Result<(), String> {
        let size = window.inner_size();
        let surface = self
            .instance
            .create_surface(window.clone())
            .map_err(|error| format!("failed to create output-window surface: {error}"))?;

        let caps = surface.get_capabilities(&self.adapter);
        if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            return Err(
                "output-window surface does not support render attachments on the active GPU"
                    .to_string(),
            );
        }
        let format = preferred_surface_format(&caps.formats).ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no texture formats)"
                .to_string()
        })?;
        let alpha_mode = caps.alpha_modes.first().copied().ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no alpha modes)".to_string()
        })?;
        let present_mode = preferred_present_mode(&caps.present_modes).ok_or_else(|| {
            "output-window surface is incompatible with the active GPU (no present modes)"
                .to_string()
        })?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        // Blit pipeline targeting this surface's format.
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit BGL"),
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

        let vertex = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Vertex"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
            });
        let fragment = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Fragment"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blit PL"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Blit Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
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

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.output = Some(OutputTarget {
            window,
            surface,
            config,
            pipeline,
            bind_group,
        });
        Ok(())
    }

    /// Start a new patch/visual generation without sampling any temporal or
    /// asynchronous readback data produced by the prior generation.
    ///
    /// The application must also advance its external visual epoch and clear
    /// any CPU-side NTSC-presented frame. Pending map callbacks may complete,
    /// but their sequence numbers are at or below this invalidation watermark
    /// and `poll_readback` will recycle them without returning their pixels.
    pub fn reset_visual_generation(&mut self) {
        self.temporal_state.reset();
        self.last_harvested_readback_sequence = self
            .last_harvested_readback_sequence
            .max(self.next_readback_sequence.saturating_sub(1));
    }

    /// Blackout: clear the final composite to black. Everything downstream
    /// — panel preview, output window, Spout, NTSC — goes dark together.
    pub fn clear_composite(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blackout"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.composite_views[0],
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
    }

    pub fn close_output(&mut self) {
        self.output = None;
    }

    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub fn output_window_id(&self) -> Option<winit::window::WindowId> {
        self.output.as_ref().map(|o| o.window.id())
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        if let Some(out) = self.output.as_mut() {
            if width > 0 && height > 0 {
                out.config.width = width;
                out.config.height = height;
                out.surface.configure(&self.device, &out.config);
            }
        }
    }

    /// Encode the letterboxed blit onto the output window's surface.
    /// Returns the surface texture to present after the encoder is
    /// submitted; None if the surface wasn't ready this frame.
    pub fn render_output(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu::SurfaceTexture> {
        let out = self.output.as_mut()?;

        let surface_texture = match out.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                out.surface.configure(&self.device, &out.config);
                return None;
            }
            _ => return None,
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Letterbox: fit the composite's aspect into the window.
        let (sw, sh) = (out.config.width as f32, out.config.height as f32);
        let aspect = self.output_width as f32 / self.output_height as f32;
        let (vw, vh) = if sw / sh > aspect {
            (sh * aspect, sh)
        } else {
            (sw, sw / aspect)
        };
        let (vx, vy) = ((sw - vw) * 0.5, (sh - vh) * 0.5);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Output Blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.set_pipeline(&out.pipeline);
        pass.set_bind_group(0, &out.bind_group, &[]);
        pass.set_viewport(vx, vy, vw.max(1.0), vh.max(1.0), 0.0, 1.0);
        pass.draw(0..3, 0..1);
        drop(pass);

        Some(surface_texture)
    }

    /// Temporal pass driven by a real elapsed time. Live rendering should pass
    /// its measured frame delta; deterministic export should pass `1.0 / fps`.
    pub fn render_temporal_with_dt(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &TemporalParams,
        delta_seconds: f32,
    ) {
        encode_temporal_with_dt(
            &self.device,
            encoder,
            params,
            &self.temporal_pipeline,
            &self.temporal_bind_group_layout,
            &self.temporal_uniform_layout,
            &self.sampler,
            &self.composite_textures,
            &self.composite_views,
            &self.history_texture,
            &self.history_view,
            &self.feedback_texture,
            &self.feedback_view,
            &mut self.temporal_state,
            delta_seconds,
            self.output_width,
            self.output_height,
        );
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Force reconfigure the surface (e.g. after Lost/Outdated during fullscreen transition).
    pub fn reconfigure_surface(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Render all layers composited together. Final result ends up in composite_views[0].
    /// `mods` is aligned with `layers` and carries each layer's modulated
    /// effect uniforms and opacity (bases stay untouched on the Layer).
    pub fn render_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
        mods: &[(EffectUniforms, f32)],
    ) {
        // Render in reverse order: last layer in the vec is the bottom,
        // first layer (index 0, "Layer 1" in UI) ends up on top.
        let visible_layers: Vec<(&Layer, &(EffectUniforms, f32))> = layers
            .iter()
            .zip(mods.iter())
            .filter(|(l, _)| l.visible)
            .rev()
            .collect();

        // Visibility is a transport control, not just a compositing hint. If
        // every layer is hidden, clear the prior accumulation instead of
        // leaving the last visible frame latched on every output.
        if visible_layers.is_empty() {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[0],
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

        for (i, (layer, (uniforms, opacity))) in visible_layers.iter().enumerate() {
            let uniforms = *uniforms;

            // Each pass needs its own buffer because queue.write_buffer writes
            // all execute before the encoder's render passes run on the GPU.
            let fx_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Layer FX Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Layer FX"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[1],
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
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &tex_bg, &[]);
                pass.set_bind_group(1, &uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // Step 2: Composite layer onto accumulated result.
            // The bottom layer composites onto a cleared black base with its
            // (modulated) opacity — it was previously copied raw, which
            // silently ignored opacity whenever only one layer was visible.
            // Its blend mode is forced to Normal: screen against black is
            // identity and multiply is a black frame — only surprises.
            if i == 0 {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Clear Base"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[0],
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
            }
            {
                // Composite base[0] + overlay[1] → temp[2]
                let comp_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Composite Uniforms"),
                            contents: bytemuck::cast_slice(&[CompositeUniforms {
                                opacity: *opacity,
                                blend_mode: if i == 0 { 0 } else { layer.blend_mode.as_u32() },
                                _pad: [0.0; 2],
                            }]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let composite_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite Textures BG"),
                    layout: &self.composite_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                let composite_uniform_bg =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Composite Uniform BG"),
                        layout: &self.composite_uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: comp_buffer.as_entire_binding(),
                        }],
                    });

                // Render composite to [2]
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Composite Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.composite_views[2],
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
                    pass.set_pipeline(&self.composite_pipeline);
                    pass.set_bind_group(0, &composite_tex_bg, &[]);
                    pass.set_bind_group(1, &composite_uniform_bg, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Copy [2] → [0] so it becomes the new accumulated base
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[2],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[0],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.output_width,
                        height: self.output_height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }

    /// Apply master effects to the final composite (already in [0]).
    /// Reads [0], applies effects → [2], copies back to [0].
    pub fn render_master_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        master_uniforms: &EffectUniforms,
    ) {
        let fx_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Master FX Uniforms"),
                contents: bytemuck::cast_slice(&[*master_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Input"),
            layout: &self.effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Uniforms BG"),
            layout: &self.effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        // Render effects from [0] → [2]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Master FX Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[2],
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
            pass.set_pipeline(&self.effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy [2] → [0]
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Row stride aligned to wgpu's 256-byte copy requirement.
    fn readback_bytes_per_row(&self) -> u32 {
        (self.output_width * 4 + 255) & !255
    }

    /// Phase 1 of async readback: encode a copy of composite_textures[0]
    /// into a free staging buffer. Returns the slot index to pass to
    /// `map_readback` after the encoder is submitted, or None if every
    /// slot still has a copy in flight.
    pub fn begin_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        epoch: u64,
    ) -> Option<usize> {
        let buffer_size = (self.readback_bytes_per_row() * self.output_height) as u64;

        let idx = match self
            .readback_slots
            .iter()
            .position(|s| s.status.load(Ordering::Acquire) == SLOT_IDLE)
        {
            Some(idx) => idx,
            None if self.readback_slots.len() < MAX_READBACK_SLOTS => {
                self.readback_slots.push(ReadbackSlot {
                    buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("NTSC Readback Slot"),
                        size: buffer_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                    sequence: 0,
                    epoch,
                });
                self.readback_slots.len() - 1
            }
            None => return None,
        };

        let sequence = self.next_readback_sequence;
        self.next_readback_sequence = self.next_readback_sequence.saturating_add(1);
        self.readback_slots[idx].sequence = sequence;
        self.readback_slots[idx].epoch = epoch;
        let slot = &self.readback_slots[idx];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.readback_bytes_per_row()),
                    rows_per_image: Some(self.output_height),
                },
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
        slot.status.store(SLOT_MAP_PENDING, Ordering::Release);
        Some(idx)
    }

    /// Phase 2: request the async map. Must be called after the encoder
    /// from `begin_readback` has been submitted. Never blocks.
    pub fn map_readback(&self, idx: usize) {
        let slot = &self.readback_slots[idx];
        let status = slot.status.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let new = if result.is_ok() {
                    SLOT_MAPPED
                } else {
                    SLOT_MAP_FAILED
                };
                status.store(new, Ordering::Release);
            });
    }

    /// Phase 3: harvest a completed readback if any. Non-blocking; returns
    /// the freshest completed frame. Completed or subsequently completing
    /// older slots are discarded, so callback reordering can never make an
    /// old NTSC/Spout frame overwrite a newer composite.
    pub fn poll_readback(&mut self) -> Option<ReadbackFrame> {
        if self.readback_slots.is_empty() {
            return None;
        }
        // Drive map callbacks without waiting.
        let _ = self.device.poll(wgpu::PollType::Poll);

        let newest_idx = self
            .readback_slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.status.load(Ordering::Acquire) == SLOT_MAPPED
                    && slot.sequence > self.last_harvested_readback_sequence
            })
            .max_by_key(|(_, slot)| slot.sequence)
            .map(|(idx, _)| idx);
        let newest_sequence = newest_idx.map(|idx| self.readback_slots[idx].sequence);

        let row_bytes = (self.output_width * 4) as usize;
        let padded_row = self.readback_bytes_per_row() as usize;
        let h = self.output_height as usize;
        let mut harvested: Option<ReadbackFrame> = None;
        for (idx, slot) in self.readback_slots.iter().enumerate() {
            match slot.status.load(Ordering::Acquire) {
                SLOT_MAPPED if Some(idx) == newest_idx => {
                    let data = slot.buffer.slice(..).get_mapped_range();
                    let mut pixels = Vec::with_capacity(row_bytes * h);
                    for row in 0..h {
                        let start = row * padded_row;
                        pixels.extend_from_slice(&data[start..start + row_bytes]);
                    }
                    drop(data);
                    slot.buffer.unmap();
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                    harvested = Some(ReadbackFrame {
                        pixels,
                        epoch: slot.epoch,
                    });
                }
                SLOT_MAPPED => {
                    // This completed map is older than the freshest mapped
                    // frame (or than one already returned on an earlier poll).
                    slot.buffer.unmap();
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                }
                SLOT_MAP_FAILED => {
                    // Device hiccup (e.g. surface loss); recycle the slot.
                    slot.status.store(SLOT_IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
        if let Some(sequence) = newest_sequence {
            self.last_harvested_readback_sequence = sequence;
        }
        harvested
    }

    /// Write RGBA pixels back to composite_textures[0].
    pub fn write_composite(&self, pixels: &[u8]) {
        let w = self.output_width;
        let h = self.output_height;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
}
