//! Fixed-allocation RGBA16Float primitives for the advanced composition GPU.
//!
//! Preparation owns every texture, pipeline, bind group, and uniform arena.
//! The encode methods only update existing buffers and record commands, so a
//! warmed frame cannot accidentally turn topology into allocation work.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::effects::params::TemporalParams;
use crate::layers::BlendMode;
use crate::renderer::blend::{composite_shader_source, matte_composite_shader_source};
use crate::renderer::compositor::MatteCompositeUniforms;
use crate::spatial::EffectPassUniforms;
use crate::temporal::{
    TemporalFrameAction, TemporalFrameInput, TemporalGpuUniforms, TemporalOriginalsGpuUniforms,
    TemporalResetCause, TemporalState, TemporalStateMetrics, TEMPORAL_HISTORY_LEN,
};
use crate::visual_rack::{
    CreativeResourcePlan, ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS,
    ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS, MAX_CREATIVE_GPU_BYTES,
};

pub(crate) const HOST_WORKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const HOST_PRESENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub(crate) const HOST_SURFACE_COUNT: usize = 6;
pub(crate) const MAX_HOST_UNIFORM_SLOTS: usize = 16_384;
const HISTORY_LEN: u32 = TEMPORAL_HISTORY_LEN;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostCapacities {
    pub effect_slots: usize,
    pub composite_slots: usize,
    pub matte_slots: usize,
    /// Allocate the double-buffered full-frame N-1 Program producer only when
    /// the immutable graph declares at least one previous-frame consumer.
    pub retain_program_history: bool,
    /// The already-validated whole-plan ledger. Host construction refuses to
    /// allocate unless this includes every host-owned working/history surface.
    pub resources: CreativeResourcePlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum HostSurface {
    Ping = 0,
    Pong = 1,
    A = 2,
    B = 3,
    Program = 4,
    GroupScratch = 5,
}

impl HostSurface {
    const ALL: [Self; HOST_SURFACE_COUNT] = [
        Self::Ping,
        Self::Pong,
        Self::A,
        Self::B,
        Self::Program,
        Self::GroupScratch,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Ping => "Advanced composition ping",
            Self::Pong => "Advanced composition pong",
            Self::A => "Advanced composition A bus",
            Self::B => "Advanced composition B bus",
            Self::Program => "Advanced composition Program bus",
            Self::GroupScratch => "Advanced composition group scratch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostUniformSlot(pub u32);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HostAllocationSnapshot {
    pub textures: u64,
    pub texture_views: u64,
    pub buffers: u64,
    pub samplers: u64,
    pub bind_group_layouts: u64,
    pub bind_groups: u64,
    pub shader_modules: u64,
    pub pipeline_layouts: u64,
    pub pipelines: u64,
}

impl HostAllocationSnapshot {
    pub const fn total(self) -> u64 {
        self.textures
            + self.texture_views
            + self.buffers
            + self.samplers
            + self.bind_group_layouts
            + self.bind_groups
            + self.shader_modules
            + self.pipeline_layouts
            + self.pipelines
    }
}

#[derive(Default)]
struct HostAllocationCounters {
    textures: AtomicU64,
    texture_views: AtomicU64,
    buffers: AtomicU64,
    samplers: AtomicU64,
    bind_group_layouts: AtomicU64,
    bind_groups: AtomicU64,
    shader_modules: AtomicU64,
    pipeline_layouts: AtomicU64,
    pipelines: AtomicU64,
}

impl HostAllocationCounters {
    fn snapshot(&self) -> HostAllocationSnapshot {
        HostAllocationSnapshot {
            textures: self.textures.load(Ordering::Relaxed),
            texture_views: self.texture_views.load(Ordering::Relaxed),
            buffers: self.buffers.load(Ordering::Relaxed),
            samplers: self.samplers.load(Ordering::Relaxed),
            bind_group_layouts: self.bind_group_layouts.load(Ordering::Relaxed),
            bind_groups: self.bind_groups.load(Ordering::Relaxed),
            shader_modules: self.shader_modules.load(Ordering::Relaxed),
            pipeline_layouts: self.pipeline_layouts.load(Ordering::Relaxed),
            pipelines: self.pipelines.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompositionHostError {
    ZeroDimensions([u32; 2]),
    DimensionsExceedDevice {
        requested: [u32; 2],
        limit: u32,
    },
    UniformCapacity {
        requested: usize,
        limit: usize,
    },
    UniformArenaOverflow,
    UniformSlotOutOfRange {
        slot: u32,
        capacity: usize,
    },
    ResourcePlanDimensions {
        planned: [u32; 2],
        requested: [u32; 2],
    },
    ResourcePlanUnderreports {
        planned_rgba16: u32,
        required_rgba16: u32,
        planned_compat8: u32,
        required_compat8: u32,
    },
    ResourcePlanBytes {
        declared: u64,
        calculated: u64,
        limit: u64,
    },
    PresentFormat(wgpu::TextureFormat),
    GpuInitialization(String),
}

impl fmt::Display for CompositionHostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions(size) => write!(
                formatter,
                "composition host dimensions must be nonzero, got {}x{}",
                size[0], size[1]
            ),
            Self::DimensionsExceedDevice { requested, limit } => write!(
                formatter,
                "composition host dimensions {}x{} exceed device limit {limit}",
                requested[0], requested[1]
            ),
            Self::UniformCapacity { requested, limit } => write!(
                formatter,
                "composition host requests {requested} uniform slots; limit is {limit}"
            ),
            Self::UniformArenaOverflow => {
                formatter.write_str("composition host uniform arena size overflowed")
            }
            Self::UniformSlotOutOfRange { slot, capacity } => write!(
                formatter,
                "composition host uniform slot {slot} is outside capacity {capacity}"
            ),
            Self::ResourcePlanDimensions { planned, requested } => write!(
                formatter,
                "composition host resource plan is {}x{}, requested {}x{}",
                planned[0], planned[1], requested[0], requested[1]
            ),
            Self::ResourcePlanUnderreports {
                planned_rgba16,
                required_rgba16,
                planned_compat8,
                required_compat8,
            } => write!(
                formatter,
                "composition host resource plan under-reports warm surfaces: RGBA16 {planned_rgba16}/{required_rgba16}, Compat8 {planned_compat8}/{required_compat8}"
            ),
            Self::ResourcePlanBytes {
                declared,
                calculated,
                limit,
            } => write!(
                formatter,
                "composition host resource plan declares {declared} bytes; format ledger calculates {calculated} (limit {limit})"
            ),
            Self::PresentFormat(format) => write!(
                formatter,
                "composition host cannot present into {format:?}; expected {HOST_PRESENT_FORMAT:?}"
            ),
            Self::GpuInitialization(error) => {
                write!(
                    formatter,
                    "composition host GPU initialization failed: {error}"
                )
            }
        }
    }
}

impl std::error::Error for CompositionHostError {}

struct HostSurfaceResource {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub(crate) struct HostSurfaceRef<'a> {
    pub texture: &'a wgpu::Texture,
    pub view: &'a wgpu::TextureView,
}

struct HostUniformArena {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    stride: u64,
    item_size: u64,
    capacity: usize,
}

impl HostUniformArena {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        item_size: u64,
        capacity: usize,
        label: &'static str,
        counters: &HostAllocationCounters,
    ) -> Result<Self, CompositionHostError> {
        if capacity > MAX_HOST_UNIFORM_SLOTS {
            return Err(CompositionHostError::UniformCapacity {
                requested: capacity,
                limit: MAX_HOST_UNIFORM_SLOTS,
            });
        }
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let stride = item_size
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or(CompositionHostError::UniformArenaOverflow)?;
        let physical_capacity = capacity.max(1);
        let byte_len = stride
            .checked_mul(physical_capacity as u64)
            .ok_or(CompositionHostError::UniformArenaOverflow)?;
        let last_offset = stride
            .checked_mul(physical_capacity.saturating_sub(1) as u64)
            .ok_or(CompositionHostError::UniformArenaOverflow)?;
        if byte_len > device.limits().max_buffer_size || last_offset > u64::from(u32::MAX) {
            return Err(CompositionHostError::UniformArenaOverflow);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        counters.buffers.fetch_add(1, Ordering::Relaxed);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: NonZeroU64::new(item_size),
                }),
            }],
        });
        counters.bind_groups.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            buffer,
            bind_group,
            stride,
            item_size,
            capacity,
        })
    }

    fn offset(&self, slot: HostUniformSlot) -> Result<u32, CompositionHostError> {
        if slot.0 as usize >= self.capacity {
            return Err(CompositionHostError::UniformSlotOutOfRange {
                slot: slot.0,
                capacity: self.capacity,
            });
        }
        u32::try_from(self.stride * u64::from(slot.0))
            .map_err(|_| CompositionHostError::UniformArenaOverflow)
    }

    fn write<T: bytemuck::Pod>(
        &self,
        queue: &wgpu::Queue,
        slot: HostUniformSlot,
        value: &T,
    ) -> Result<(), CompositionHostError> {
        debug_assert_eq!(std::mem::size_of::<T>() as u64, self.item_size);
        let offset = self.offset(slot)?;
        queue.write_buffer(&self.buffer, u64::from(offset), bytemuck::bytes_of(value));
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HostCompositeUniforms {
    pub opacity: f32,
    pub blend_mode: u32,
    pub _pad: [u32; 2],
}

impl HostCompositeUniforms {
    pub fn new(opacity: f32, blend_mode: BlendMode) -> Self {
        Self {
            opacity: if opacity.is_finite() {
                opacity.clamp(0.0, 1.0)
            } else {
                1.0
            },
            blend_mode: blend_mode.as_u32(),
            _pad: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HostBusUniforms {
    crossfade: f32,
    _pad: [f32; 3],
}

/// Dedicated post-temporal Refresh Garden ABI. Keeping this independent from
/// the frozen TemporalOriginals ABI lets the pre-Garden temporal pass upload a
/// literal zero Garden amount while this pass retains the staged recurrence.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HostRoutedGardenUniforms {
    /// feedback zoom, feedback rotation degrees, feedback validity, reserved
    feedback_values: [f32; 4],
    /// amount, threshold, softness, decay
    garden_values: [f32; 4],
    /// observation ticks, packed runtime, gate code, reserved
    garden_modes: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<HostCompositeUniforms>() == 16);
const _: () = assert!(std::mem::size_of::<HostBusUniforms>() == 16);
const _: () = assert!(std::mem::size_of::<HostRoutedGardenUniforms>() == 48);

pub(crate) struct HostEffectSource {
    bind_group: wgpu::BindGroup,
}

pub(crate) struct HostTextureInputs {
    bind_group: wgpu::BindGroup,
}

pub(crate) struct HostCompositeInputs {
    bind_group: wgpu::BindGroup,
}

pub(crate) struct HostMatteInputs {
    bind_group: wgpu::BindGroup,
}

pub(crate) struct HostTemporalInput {
    bind_group: wgpu::BindGroup,
    current_copy_group: wgpu::BindGroup,
    output_copy_group: wgpu::BindGroup,
}

pub(crate) struct HostRoutedGardenInput {
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HostFrameTiming {
    temporal: TemporalFrameInput,
}

impl HostFrameTiming {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "legacy dt/pause host adapter is retained for compatibility goldens"
        )
    )]
    pub fn new(delta_seconds: f32, advance_program: bool) -> Self {
        Self {
            temporal: TemporalFrameInput::legacy(delta_seconds, advance_program),
        }
    }

    pub const fn from_temporal_input(temporal: TemporalFrameInput) -> Self {
        Self { temporal }
    }

    pub const fn temporal_input(self) -> TemporalFrameInput {
        self.temporal
    }
}

pub(crate) struct CompositionHost {
    dimensions: [u32; 2],
    surfaces: [HostSurfaceResource; HOST_SURFACE_COUNT],
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    effect_texture_layout: wgpu::BindGroupLayout,
    universal_texture_layout: wgpu::BindGroupLayout,
    composite_texture_layout: wgpu::BindGroupLayout,
    matte_texture_layout: wgpu::BindGroupLayout,
    temporal_texture_layout: wgpu::BindGroupLayout,
    routed_garden_texture_layout: wgpu::BindGroupLayout,
    effect_pipeline: wgpu::RenderPipeline,
    copy_pipeline: wgpu::RenderPipeline,
    compat8_copy_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    matte_pipeline: wgpu::RenderPipeline,
    bus_pipeline: wgpu::RenderPipeline,
    temporal_pipeline: wgpu::RenderPipeline,
    temporal_originals_pipeline: wgpu::RenderPipeline,
    routed_garden_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    effect_uniforms: HostUniformArena,
    composite_uniforms: HostUniformArena,
    matte_uniforms: HostUniformArena,
    bus_uniform_buffer: wgpu::Buffer,
    bus_uniform_group: wgpu::BindGroup,
    temporal_uniform_buffer: wgpu::Buffer,
    temporal_uniform_group: wgpu::BindGroup,
    temporal_originals_uniform_buffer: wgpu::Buffer,
    temporal_originals_uniform_group: wgpu::BindGroup,
    routed_garden_uniform_buffer: wgpu::Buffer,
    routed_garden_uniform_group: wgpu::BindGroup,
    _temporal_history_texture: wgpu::Texture,
    temporal_history_view: wgpu::TextureView,
    temporal_history_layer_views: Box<[wgpu::TextureView]>,
    /// Storage of the 25-layer temporal class. `Compat8` in every production
    /// host; `Rgba16Float` only under the measurement-only Gate 6 candidate.
    temporal_history_storage: crate::precision::SurfaceStorage,
    _temporal_feedback_texture: wgpu::Texture,
    temporal_feedback_view: wgpu::TextureView,
    temporal_feedback_copy_group: wgpu::BindGroup,
    temporal_state: RefCell<TemporalState>,
    program_history: Option<[HostSurfaceResource; 2]>,
    program_history_read_index: Cell<usize>,
    program_history_initialized: Cell<bool>,
    program_history_staged: Cell<bool>,
    allocations: HostAllocationCounters,
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_layout_entry(size: u64, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    uniform_layout_entry_at(0, size, dynamic)
}

fn uniform_layout_entry_at(binding: u32, size: u64, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn create_layout(
    device: &wgpu::Device,
    label: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
    counters: &HostAllocationCounters,
) -> wgpu::BindGroupLayout {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    counters.bind_group_layouts.fetch_add(1, Ordering::Relaxed);
    layout
}

fn create_shader(
    device: &wgpu::Device,
    label: &'static str,
    source: impl Into<wgpu::ShaderSource<'static>>,
    counters: &HostAllocationCounters,
) -> wgpu::ShaderModule {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: source.into(),
    });
    counters.shader_modules.fetch_add(1, Ordering::Relaxed);
    module
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    label: &'static str,
    layouts: &[&wgpu::BindGroupLayout],
    counters: &HostAllocationCounters,
) -> wgpu::PipelineLayout {
    let bind_group_layouts = layouts.iter().copied().map(Some).collect::<Vec<_>>();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bind_group_layouts,
        immediate_size: 0,
    });
    counters.pipeline_layouts.fetch_add(1, Ordering::Relaxed);
    layout
}

#[allow(
    clippy::too_many_arguments,
    reason = "pipeline construction keeps each audited wgpu descriptor input explicit"
)]
fn create_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::PipelineLayout,
    vertex: &wgpu::ShaderModule,
    fragment: &wgpu::ShaderModule,
    entry_point: &'static str,
    format: wgpu::TextureFormat,
    counters: &HostAllocationCounters,
) -> wgpu::RenderPipeline {
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fragment,
            entry_point: Some(entry_point),
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
    counters.pipelines.fetch_add(1, Ordering::Relaxed);
    pipeline
}

fn create_surface(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    label: &'static str,
    counters: &HostAllocationCounters,
) -> HostSurfaceResource {
    create_surface_with_format(device, dimensions, label, HOST_WORKING_FORMAT, counters)
}

fn create_surface_with_format(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    label: &'static str,
    format: wgpu::TextureFormat,
    counters: &HostAllocationCounters,
) -> HostSurfaceResource {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    counters.textures.fetch_add(1, Ordering::Relaxed);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    counters.texture_views.fetch_add(1, Ordering::Relaxed);
    HostSurfaceResource { texture, view }
}

fn create_fixed_uniform<T: bytemuck::Pod + bytemuck::Zeroable>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    counters: &HostAllocationCounters,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let size = std::mem::size_of::<T>() as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    counters.buffers.fetch_add(1, Ordering::Relaxed);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    counters.bind_groups.fetch_add(1, Ordering::Relaxed);
    (buffer, bind_group)
}

fn create_temporal_originals_uniform(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    legacy: &wgpu::Buffer,
    counters: &HostAllocationCounters,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let originals = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Advanced composition temporal originals uniform"),
        size: std::mem::size_of::<TemporalOriginalsGpuUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    counters.buffers.fetch_add(1, Ordering::Relaxed);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Advanced composition temporal uniforms"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: legacy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: originals.as_entire_binding(),
            },
        ],
    });
    counters.bind_groups.fetch_add(1, Ordering::Relaxed);
    (originals, bind_group)
}

fn validate_host_resource_plan(
    dimensions: [u32; 2],
    capacities: HostCapacities,
    history_storage: crate::precision::SurfaceStorage,
) -> Result<(), CompositionHostError> {
    let resources = capacities.resources;
    if resources.output_size != dimensions {
        return Err(CompositionHostError::ResourcePlanDimensions {
            planned: resources.output_size,
            requested: dimensions,
        });
    }
    let required_rgba16 = HOST_SURFACE_COUNT as u32
        + u32::from(capacities.retain_program_history)
            * (1 + ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS);
    let required_compat8 = ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS;
    if resources.rgba16_surface_layers < required_rgba16
        || resources.compat8_surface_layers < required_compat8
    {
        return Err(CompositionHostError::ResourcePlanUnderreports {
            planned_rgba16: resources.rgba16_surface_layers,
            required_rgba16,
            planned_compat8: resources.compat8_surface_layers,
            required_compat8,
        });
    }
    let pixels = u64::from(dimensions[0]).saturating_mul(u64::from(dimensions[1]));
    // The `compat8_surface_layers` class is the temporal-history class: 25
    // layers at the host's history storage width. The settled path stores
    // them Compat8 at 4 bytes; the evaluation-only Full-16 constructor
    // widens exactly this class to 8 and the plan must charge the truth.
    let calculated_bytes = pixels
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_mul(u64::from(resources.rgba16_surface_layers)))
        .and_then(|rgba16| {
            pixels
                .checked_mul(history_storage.bytes_per_pixel())
                .and_then(|bytes| bytes.checked_mul(u64::from(resources.compat8_surface_layers)))
                .and_then(|compat8| rgba16.checked_add(compat8))
        })
        .unwrap_or(u64::MAX);
    let declared_layers = resources
        .rgba16_surface_layers
        .checked_add(resources.compat8_surface_layers);
    if declared_layers != Some(resources.retained_surface_layers)
        || resources.creative_bytes != calculated_bytes
        || resources.creative_bytes > MAX_CREATIVE_GPU_BYTES
    {
        return Err(CompositionHostError::ResourcePlanBytes {
            declared: resources.creative_bytes,
            calculated: calculated_bytes,
            limit: MAX_CREATIVE_GPU_BYTES,
        });
    }
    Ok(())
}

impl CompositionHost {
    /// The production constructor: the settled
    /// `AdvancedWorking16HistoryCompat8` path, byte-identical to what it has
    /// always built. Every live and export executor comes through here.
    pub fn new(
        device: &wgpu::Device,
        dimensions: [u32; 2],
        capacities: HostCapacities,
    ) -> Result<Self, CompositionHostError> {
        Self::new_with_history_storage(
            device,
            dimensions,
            capacities,
            crate::precision::SurfaceStorage::Compat8,
        )
    }

    /// The Gate 6 evaluation constructor. `history_storage` selects the
    /// storage of exactly the 25-layer temporal class — the 24-layer clean
    /// history ring and the recursive feedback image — leaving working,
    /// present, and every other surface untouched. `Compat8` is the settled
    /// default; `Rgba16Float` is the `ExperimentalFull16History` candidate,
    /// which is **measurement-only**: no production call site constructs it,
    /// no wire action or patch field selects it, and the settled default does
    /// not move. Because the ring and feedback are written exclusively by
    /// render passes (the no-dither conversion pipeline) and read exclusively
    /// by `textureLoad`/`textureSample`, both storages present identical
    /// *linear* values to every consumer — sRGB8 encodes on write and decodes
    /// on read, f16 carries linear directly — so the candidate changes
    /// quantization, never value domain, and no consumer shader changes.
    pub fn new_with_history_storage(
        device: &wgpu::Device,
        dimensions: [u32; 2],
        capacities: HostCapacities,
        history_storage: crate::precision::SurfaceStorage,
    ) -> Result<Self, CompositionHostError> {
        let history_format = match history_storage {
            crate::precision::SurfaceStorage::Compat8 => HOST_PRESENT_FORMAT,
            crate::precision::SurfaceStorage::Rgba16Float => HOST_WORKING_FORMAT,
        };
        if dimensions.contains(&0) {
            return Err(CompositionHostError::ZeroDimensions(dimensions));
        }
        let limit = device.limits().max_texture_dimension_2d;
        if dimensions[0] > limit || dimensions[1] > limit {
            return Err(CompositionHostError::DimensionsExceedDevice {
                requested: dimensions,
                limit,
            });
        }
        for requested in [
            capacities.effect_slots,
            capacities.composite_slots,
            capacities.matte_slots,
        ] {
            if requested > MAX_HOST_UNIFORM_SLOTS {
                return Err(CompositionHostError::UniformCapacity {
                    requested,
                    limit: MAX_HOST_UNIFORM_SLOTS,
                });
            }
        }
        validate_host_resource_plan(dimensions, capacities, history_storage)?;

        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let counters = HostAllocationCounters::default();

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Advanced composition linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Advanced composition nearest sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        counters.samplers.fetch_add(2, Ordering::Relaxed);

        let effect_texture_layout = create_layout(
            device,
            "Advanced composition effects textures",
            &[sampled_texture_entry(0), sampler_entry(1), sampler_entry(2)],
            &counters,
        );
        let universal_texture_layout = create_layout(
            device,
            "Advanced composition copy/bus textures",
            &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                sampler_entry(3),
            ],
            &counters,
        );
        let composite_texture_layout = create_layout(
            device,
            "Advanced composition composite textures",
            &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                sampler_entry(2),
            ],
            &counters,
        );
        let matte_texture_layout = create_layout(
            device,
            "Advanced composition matte textures",
            &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                sampler_entry(3),
            ],
            &counters,
        );
        let temporal_texture_layout = create_layout(
            device,
            "Advanced composition temporal textures",
            &[
                sampled_texture_entry(0),
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
                sampler_entry(2),
                sampled_texture_entry(3),
            ],
            &counters,
        );
        let routed_garden_texture_layout = create_layout(
            device,
            "Advanced composition routed Garden textures",
            &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                sampler_entry(3),
            ],
            &counters,
        );
        let effect_uniform_layout = create_layout(
            device,
            "Advanced composition effects uniform arena",
            &[uniform_layout_entry(
                std::mem::size_of::<EffectPassUniforms>() as u64,
                true,
            )],
            &counters,
        );
        let composite_uniform_layout = create_layout(
            device,
            "Advanced composition composite uniform arena",
            &[uniform_layout_entry(
                std::mem::size_of::<HostCompositeUniforms>() as u64,
                true,
            )],
            &counters,
        );
        let matte_uniform_layout = create_layout(
            device,
            "Advanced composition matte uniform arena",
            &[uniform_layout_entry(
                std::mem::size_of::<MatteCompositeUniforms>() as u64,
                true,
            )],
            &counters,
        );
        let bus_uniform_layout = create_layout(
            device,
            "Advanced composition bus uniform",
            &[uniform_layout_entry(
                std::mem::size_of::<HostBusUniforms>() as u64,
                false,
            )],
            &counters,
        );
        let temporal_uniform_layout = create_layout(
            device,
            "Advanced composition legacy temporal uniform",
            &[uniform_layout_entry(
                std::mem::size_of::<TemporalGpuUniforms>() as u64,
                false,
            )],
            &counters,
        );
        let temporal_originals_uniform_layout = create_layout(
            device,
            "Advanced composition temporal originals uniforms",
            &[
                uniform_layout_entry_at(
                    0,
                    std::mem::size_of::<TemporalGpuUniforms>() as u64,
                    false,
                ),
                uniform_layout_entry_at(
                    1,
                    std::mem::size_of::<TemporalOriginalsGpuUniforms>() as u64,
                    false,
                ),
            ],
            &counters,
        );
        let routed_garden_uniform_layout = create_layout(
            device,
            "Advanced composition routed Garden uniform",
            &[uniform_layout_entry(
                std::mem::size_of::<HostRoutedGardenUniforms>() as u64,
                false,
            )],
            &counters,
        );

        let vertex = create_shader(
            device,
            "Advanced composition fullscreen vertex",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
            &counters,
        );
        let effects = create_shader(
            device,
            "Advanced composition effects",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/effects.wgsl").into()),
            &counters,
        );
        let composite = create_shader(
            device,
            "Advanced composition blend",
            wgpu::ShaderSource::Wgsl(composite_shader_source()),
            &counters,
        );
        let matte = create_shader(
            device,
            "Advanced composition matte",
            wgpu::ShaderSource::Wgsl(matte_composite_shader_source()),
            &counters,
        );
        let host = create_shader(
            device,
            "Advanced composition host",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/composition_host.wgsl").into()),
            &counters,
        );
        let temporal = create_shader(
            device,
            "Advanced composition temporal",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/temporal.wgsl").into()),
            &counters,
        );
        let temporal_originals = create_shader(
            device,
            "Advanced composition temporal originals",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/temporal_originals.wgsl").into()),
            &counters,
        );
        let routed_garden = create_shader(
            device,
            "Advanced composition routed Garden",
            wgpu::ShaderSource::Wgsl(include_str!("../shaders/refresh_garden_routed.wgsl").into()),
            &counters,
        );

        let effect_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition effects pipeline layout",
            &[&effect_texture_layout, &effect_uniform_layout],
            &counters,
        );
        let universal_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition copy/present pipeline layout",
            &[&universal_texture_layout],
            &counters,
        );
        let composite_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition composite pipeline layout",
            &[&composite_texture_layout, &composite_uniform_layout],
            &counters,
        );
        let matte_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition matte pipeline layout",
            &[&matte_texture_layout, &matte_uniform_layout],
            &counters,
        );
        let bus_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition bus pipeline layout",
            &[&universal_texture_layout, &bus_uniform_layout],
            &counters,
        );
        let temporal_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition temporal pipeline layout",
            &[&temporal_texture_layout, &temporal_uniform_layout],
            &counters,
        );
        let temporal_originals_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition temporal originals pipeline layout",
            &[&temporal_texture_layout, &temporal_originals_uniform_layout],
            &counters,
        );
        let routed_garden_pipeline_layout = create_pipeline_layout(
            device,
            "Advanced composition routed Garden pipeline layout",
            &[&routed_garden_texture_layout, &routed_garden_uniform_layout],
            &counters,
        );

        let effect_pipeline = create_pipeline(
            device,
            "Advanced composition effects pipeline",
            &effect_pipeline_layout,
            &vertex,
            &effects,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let copy_pipeline = create_pipeline(
            device,
            "Advanced composition copy pipeline",
            &universal_pipeline_layout,
            &vertex,
            &host,
            "fs_copy",
            HOST_WORKING_FORMAT,
            &counters,
        );
        // The no-dither history conversion pipeline targets the temporal
        // class's own storage. Under the settled Compat8 path this is the
        // byte-identical sRGB8 target it has always been; under the Full-16
        // candidate the same shader writes linear f16, so the only change is
        // quantization at the target.
        let compat8_copy_pipeline = create_pipeline(
            device,
            "Advanced composition no-dither Compat8 copy pipeline",
            &universal_pipeline_layout,
            &vertex,
            &host,
            "fs_copy",
            history_format,
            &counters,
        );
        let present_pipeline = create_pipeline(
            device,
            "Advanced composition present pipeline",
            &universal_pipeline_layout,
            &vertex,
            &host,
            "fs_present",
            HOST_PRESENT_FORMAT,
            &counters,
        );
        let composite_pipeline = create_pipeline(
            device,
            "Advanced composition composite pipeline",
            &composite_pipeline_layout,
            &vertex,
            &composite,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let matte_pipeline = create_pipeline(
            device,
            "Advanced composition matte pipeline",
            &matte_pipeline_layout,
            &vertex,
            &matte,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let bus_pipeline = create_pipeline(
            device,
            "Advanced composition bus pipeline",
            &bus_pipeline_layout,
            &vertex,
            &host,
            "fs_bus",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let temporal_pipeline = create_pipeline(
            device,
            "Advanced composition temporal pipeline",
            &temporal_pipeline_layout,
            &vertex,
            &temporal,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let temporal_originals_pipeline = create_pipeline(
            device,
            "Advanced composition temporal originals pipeline",
            &temporal_originals_pipeline_layout,
            &vertex,
            &temporal_originals,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );
        let routed_garden_pipeline = create_pipeline(
            device,
            "Advanced composition routed Garden pipeline",
            &routed_garden_pipeline_layout,
            &vertex,
            &routed_garden,
            "fs_main",
            HOST_WORKING_FORMAT,
            &counters,
        );

        let effect_uniforms = HostUniformArena::new(
            device,
            &effect_uniform_layout,
            std::mem::size_of::<EffectPassUniforms>() as u64,
            capacities.effect_slots,
            "Advanced composition effects uniforms",
            &counters,
        )?;
        let composite_uniforms = HostUniformArena::new(
            device,
            &composite_uniform_layout,
            std::mem::size_of::<HostCompositeUniforms>() as u64,
            capacities.composite_slots,
            "Advanced composition composite uniforms",
            &counters,
        )?;
        let matte_uniforms = HostUniformArena::new(
            device,
            &matte_uniform_layout,
            std::mem::size_of::<MatteCompositeUniforms>() as u64,
            capacities.matte_slots,
            "Advanced composition matte uniforms",
            &counters,
        )?;
        let (bus_uniform_buffer, bus_uniform_group) = create_fixed_uniform::<HostBusUniforms>(
            device,
            &bus_uniform_layout,
            "Advanced composition bus uniform",
            &counters,
        );
        let (temporal_uniform_buffer, temporal_uniform_group) =
            create_fixed_uniform::<TemporalGpuUniforms>(
                device,
                &temporal_uniform_layout,
                "Advanced composition legacy temporal uniform",
                &counters,
            );
        let (temporal_originals_uniform_buffer, temporal_originals_uniform_group) =
            create_temporal_originals_uniform(
                device,
                &temporal_originals_uniform_layout,
                &temporal_uniform_buffer,
                &counters,
            );
        let (routed_garden_uniform_buffer, routed_garden_uniform_group) =
            create_fixed_uniform::<HostRoutedGardenUniforms>(
                device,
                &routed_garden_uniform_layout,
                "Advanced composition routed Garden uniform",
                &counters,
            );

        let surfaces = HostSurface::ALL
            .map(|surface| create_surface(device, dimensions, surface.label(), &counters));
        let temporal_history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Advanced composition Compat8 temporal clean history"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: HISTORY_LEN,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: history_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        counters.textures.fetch_add(1, Ordering::Relaxed);
        let temporal_history_view =
            temporal_history_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        counters.texture_views.fetch_add(1, Ordering::Relaxed);
        let temporal_history_layer_views = (0..HISTORY_LEN)
            .map(|layer| {
                let view = temporal_history_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Advanced composition Compat8 temporal clean history slice"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                counters.texture_views.fetch_add(1, Ordering::Relaxed);
                view
            })
            .collect::<Box<[_]>>();
        let HostSurfaceResource {
            texture: temporal_feedback_texture,
            view: temporal_feedback_view,
        } = create_surface_with_format(
            device,
            dimensions,
            "Advanced composition Compat8 temporal feedback",
            history_format,
            &counters,
        );
        let temporal_feedback_copy_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared Compat8 feedback copy"),
            layout: &universal_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&temporal_feedback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&temporal_feedback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&temporal_feedback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });
        counters.bind_groups.fetch_add(1, Ordering::Relaxed);
        let program_history = capacities.retain_program_history.then(|| {
            std::array::from_fn(|index| {
                create_surface(
                    device,
                    dimensions,
                    if index == 0 {
                        "Advanced composition N-1 Program history A"
                    } else {
                        "Advanced composition N-1 Program history B"
                    },
                    &counters,
                )
            })
        });

        let errors = [
            pollster::block_on(out_of_memory.pop()),
            pollster::block_on(internal.pop()),
            pollster::block_on(validation.pop()),
        ]
        .into_iter()
        .flatten()
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
        if !errors.is_empty() {
            return Err(CompositionHostError::GpuInitialization(errors.join("; ")));
        }

        Ok(Self {
            dimensions,
            surfaces,
            linear_sampler,
            nearest_sampler,
            effect_texture_layout,
            universal_texture_layout,
            composite_texture_layout,
            matte_texture_layout,
            temporal_texture_layout,
            routed_garden_texture_layout,
            effect_pipeline,
            copy_pipeline,
            compat8_copy_pipeline,
            composite_pipeline,
            matte_pipeline,
            bus_pipeline,
            temporal_pipeline,
            temporal_originals_pipeline,
            routed_garden_pipeline,
            present_pipeline,
            effect_uniforms,
            composite_uniforms,
            matte_uniforms,
            bus_uniform_buffer,
            bus_uniform_group,
            temporal_uniform_buffer,
            temporal_uniform_group,
            temporal_originals_uniform_buffer,
            temporal_originals_uniform_group,
            routed_garden_uniform_buffer,
            routed_garden_uniform_group,
            _temporal_history_texture: temporal_history_texture,
            temporal_history_storage: history_storage,
            temporal_history_view,
            temporal_history_layer_views,
            _temporal_feedback_texture: temporal_feedback_texture,
            temporal_feedback_view,
            temporal_feedback_copy_group,
            temporal_state: RefCell::new(TemporalState::default()),
            program_history,
            program_history_read_index: Cell::new(0),
            program_history_initialized: Cell::new(false),
            program_history_staged: Cell::new(false),
            allocations: counters,
        })
    }

    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    pub fn allocation_snapshot(&self) -> HostAllocationSnapshot {
        self.allocations.snapshot()
    }

    pub fn surface(&self, surface: HostSurface) -> HostSurfaceRef<'_> {
        let resource = &self.surfaces[surface.index()];
        HostSurfaceRef {
            texture: &resource.texture,
            view: &resource.view,
        }
    }

    /// Preparation-time source binding for the frozen effects shader. The
    /// returned object owns its bind group and may be reused every frame.
    pub fn prepare_effect_source(
        &self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
    ) -> HostEffectSource {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared effects source"),
            layout: &self.effect_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        HostEffectSource { bind_group }
    }

    /// Preparation-time single-source binding used by copy and presentation.
    /// The universal layout intentionally binds the same source in all three
    /// slots so its one immutable bind group can serve both entry points.
    pub fn prepare_copy_source(
        &self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
    ) -> HostTextureInputs {
        self.prepare_universal_inputs(device, source, source, source)
    }

    pub fn prepare_bus_inputs(
        &self,
        device: &wgpu::Device,
        a: &wgpu::TextureView,
        b: &wgpu::TextureView,
        program: &wgpu::TextureView,
    ) -> HostTextureInputs {
        self.prepare_universal_inputs(device, a, b, program)
    }

    fn prepare_universal_inputs(
        &self,
        device: &wgpu::Device,
        a: &wgpu::TextureView,
        b: &wgpu::TextureView,
        program: &wgpu::TextureView,
    ) -> HostTextureInputs {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared copy/bus inputs"),
            layout: &self.universal_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(program),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        HostTextureInputs { bind_group }
    }

    pub fn prepare_composite_inputs(
        &self,
        device: &wgpu::Device,
        base: &wgpu::TextureView,
        overlay: &wgpu::TextureView,
    ) -> HostCompositeInputs {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared composite inputs"),
            layout: &self.composite_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(overlay),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        HostCompositeInputs { bind_group }
    }

    pub fn prepare_matte_inputs(
        &self,
        device: &wgpu::Device,
        base: &wgpu::TextureView,
        overlay: &wgpu::TextureView,
        donor: &wgpu::TextureView,
    ) -> HostMatteInputs {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared matte inputs"),
            layout: &self.matte_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(overlay),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(donor),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        HostMatteInputs { bind_group }
    }

    pub fn prepare_temporal_input(
        &self,
        device: &wgpu::Device,
        current: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) -> HostTemporalInput {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared temporal input"),
            layout: &self.temporal_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.temporal_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.temporal_feedback_view),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        let HostTextureInputs {
            bind_group: current_copy_group,
        } = self.prepare_copy_source(device, current);
        let HostTextureInputs {
            bind_group: output_copy_group,
        } = self.prepare_copy_source(device, output);
        HostTemporalInput {
            bind_group,
            current_copy_group,
            output_copy_group,
        }
    }

    /// Preparation-time binding for the dedicated post-temporal Garden pass.
    /// `current` is the pre-Garden Pong output; feedback is host-owned and the
    /// routed signal may be either a full-frame matte image or low-res R8
    /// motion scalar. All three are fixed before warmed encoding.
    pub fn prepare_routed_garden_input(
        &self,
        device: &wgpu::Device,
        current: &wgpu::TextureView,
        signal: &wgpu::TextureView,
    ) -> HostRoutedGardenInput {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advanced composition prepared routed Garden input"),
            layout: &self.routed_garden_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.temporal_feedback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(signal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        HostRoutedGardenInput { bind_group }
    }

    pub fn write_effect_uniform(
        &self,
        queue: &wgpu::Queue,
        slot: HostUniformSlot,
        value: &EffectPassUniforms,
    ) -> Result<(), CompositionHostError> {
        let advanced = advanced_effect_pass_uniform(value);
        self.effect_uniforms.write(queue, slot, &advanced)
    }

    pub fn write_composite_uniform(
        &self,
        queue: &wgpu::Queue,
        slot: HostUniformSlot,
        value: &HostCompositeUniforms,
    ) -> Result<(), CompositionHostError> {
        self.composite_uniforms.write(queue, slot, value)
    }

    pub fn write_matte_uniform(
        &self,
        queue: &wgpu::Queue,
        slot: HostUniformSlot,
        value: &MatteCompositeUniforms,
    ) -> Result<(), CompositionHostError> {
        self.matte_uniforms.write(queue, slot, value)
    }

    pub fn encode_clear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        color: wgpu::Color,
    ) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Advanced composition clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
    }

    pub fn encode_effect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &HostEffectSource,
        target: &wgpu::TextureView,
        slot: HostUniformSlot,
    ) -> Result<(), CompositionHostError> {
        let offset = self.effect_uniforms.offset(slot)?;
        let mut pass = begin_replace_pass(encoder, "Advanced composition effects", target);
        pass.set_pipeline(&self.effect_pipeline);
        pass.set_bind_group(0, &source.bind_group, &[]);
        pass.set_bind_group(1, &self.effect_uniforms.bind_group, &[offset]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "standalone copy primitive is exposed for host GPU goldens"
        )
    )]
    pub fn encode_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &HostTextureInputs,
        target: &wgpu::TextureView,
    ) {
        let mut pass = begin_replace_pass(encoder, "Advanced composition copy", target);
        pass.set_pipeline(&self.copy_pipeline);
        pass.set_bind_group(0, &source.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &HostCompositeInputs,
        target: &wgpu::TextureView,
        slot: HostUniformSlot,
    ) -> Result<(), CompositionHostError> {
        let offset = self.composite_uniforms.offset(slot)?;
        let mut pass = begin_replace_pass(encoder, "Advanced composition composite", target);
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &inputs.bind_group, &[]);
        pass.set_bind_group(1, &self.composite_uniforms.bind_group, &[offset]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    pub fn encode_matte(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &HostMatteInputs,
        target: &wgpu::TextureView,
        slot: HostUniformSlot,
    ) -> Result<(), CompositionHostError> {
        let offset = self.matte_uniforms.offset(slot)?;
        let mut pass = begin_replace_pass(encoder, "Advanced composition matte", target);
        pass.set_pipeline(&self.matte_pipeline);
        pass.set_bind_group(0, &inputs.bind_group, &[]);
        pass.set_bind_group(1, &self.matte_uniforms.bind_group, &[offset]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    pub fn encode_bus(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &HostTextureInputs,
        target: &wgpu::TextureView,
        crossfade: f32,
    ) {
        let crossfade = if crossfade.is_finite() {
            crossfade.clamp(0.0, 1.0)
        } else {
            0.5
        };
        queue.write_buffer(
            &self.bus_uniform_buffer,
            0,
            bytemuck::bytes_of(&HostBusUniforms {
                crossfade,
                _pad: [0.0; 3],
            }),
        );
        let mut pass = begin_replace_pass(encoder, "Advanced composition bus", target);
        pass.set_pipeline(&self.bus_pipeline);
        pass.set_bind_group(0, &inputs.bind_group, &[]);
        pass.set_bind_group(1, &self.bus_uniform_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Encode the ordered LegacyTemporal marker into an arbitrary prepared
    /// RGBA16F output. `current_texture` and the output must not
    /// alias; Ping/Pong makes that invariant explicit for the caller while
    /// still allowing later rack segments to continue after this boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_temporal(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &HostTemporalInput,
        current_texture: &wgpu::Texture,
        output_texture: &wgpu::Texture,
        output_view: &wgpu::TextureView,
        params: &TemporalParams,
        timing: HostFrameTiming,
    ) {
        let wrote_current = self.encode_temporal_impl(
            queue,
            encoder,
            input,
            current_texture,
            output_texture,
            output_view,
            params,
            timing,
            None,
        );
        debug_assert!(!wrote_current);
    }

    /// Routed Garden is a post-temporal boundary. Advancing frames finish in
    /// `current_view`; Prime/Hold retain the ordinary Pong result and return
    /// `false` so the caller performs its established Pong-to-Ping copy.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_temporal_routed(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &HostTemporalInput,
        current_texture: &wgpu::Texture,
        current_view: &wgpu::TextureView,
        output_texture: &wgpu::Texture,
        output_view: &wgpu::TextureView,
        routed: &HostRoutedGardenInput,
        params: &TemporalParams,
        timing: HostFrameTiming,
    ) -> bool {
        self.encode_temporal_impl(
            queue,
            encoder,
            input,
            current_texture,
            output_texture,
            output_view,
            params,
            timing,
            Some((routed, current_view)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_temporal_impl(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &HostTemporalInput,
        current_texture: &wgpu::Texture,
        output_texture: &wgpu::Texture,
        output_view: &wgpu::TextureView,
        params: &TemporalParams,
        timing: HostFrameTiming,
        routed: Option<(&HostRoutedGardenInput, &wgpu::TextureView)>,
    ) -> bool {
        let extent = wgpu::Extent3d {
            width: self.dimensions[0],
            height: self.dimensions[1],
            depth_or_array_layers: 1,
        };
        let copy =
            |encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture, target: &wgpu::Texture| {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: source,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: target,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    extent,
                );
            };
        let render_copy = |encoder: &mut wgpu::CommandEncoder,
                           label: &'static str,
                           pipeline: &wgpu::RenderPipeline,
                           bind_group: &wgpu::BindGroup,
                           target: &wgpu::TextureView| {
            let mut pass = begin_replace_pass(encoder, label, target);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        };

        let plan = self.temporal_state.borrow_mut().stage_frame(
            params,
            timing.temporal_input(),
            self.dimensions,
        );
        match plan.action {
            TemporalFrameAction::PrimeFrozenOutput => {
                copy(encoder, current_texture, output_texture);
                render_copy(
                    encoder,
                    "Advanced composition prime Compat8 feedback",
                    &self.compat8_copy_pipeline,
                    &input.current_copy_group,
                    &self.temporal_feedback_view,
                );
                return false;
            }
            TemporalFrameAction::HoldFrozenOutput => {
                render_copy(
                    encoder,
                    "Advanced composition hold Compat8 feedback",
                    &self.copy_pipeline,
                    &self.temporal_feedback_copy_group,
                    output_view,
                );
                return false;
            }
            TemporalFrameAction::Advance { .. } => {}
        }

        if let Some((routed, current_view)) = routed {
            let mut advanced_uniforms = plan.uniforms;
            advanced_uniforms._pad = 1.0;
            queue.write_buffer(
                &self.temporal_uniform_buffer,
                0,
                bytemuck::bytes_of(&advanced_uniforms),
            );

            let mut pre_garden_originals = plan.originals_uniforms;
            pre_garden_originals.garden_values[0] = 0.0;
            let pre_garden_originals_active = pre_garden_originals.loom_values[0] > 0.0
                || pre_garden_originals.atlas_values[0] > 0.0;
            if pre_garden_originals_active {
                queue.write_buffer(
                    &self.temporal_originals_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&pre_garden_originals),
                );
                let mut pass = begin_replace_pass(
                    encoder,
                    "Advanced composition pre-Garden temporal originals",
                    output_view,
                );
                pass.set_pipeline(&self.temporal_originals_pipeline);
                pass.set_bind_group(0, &input.bind_group, &[]);
                pass.set_bind_group(1, &self.temporal_originals_uniform_group, &[]);
                pass.draw(0..3, 0..1);
            } else if plan.legacy_shader_active {
                let mut pass = begin_replace_pass(
                    encoder,
                    "Advanced composition pre-Garden temporal",
                    output_view,
                );
                pass.set_pipeline(&self.temporal_pipeline);
                pass.set_bind_group(0, &input.bind_group, &[]);
                pass.set_bind_group(1, &self.temporal_uniform_group, &[]);
                pass.draw(0..3, 0..1);
            } else {
                copy(encoder, current_texture, output_texture);
            }

            if let Some(history_write_target) = plan.history_write_target {
                render_copy(
                    encoder,
                    "Advanced composition record Compat8 clean history",
                    &self.compat8_copy_pipeline,
                    &input.current_copy_group,
                    &self.temporal_history_layer_views[history_write_target],
                );
            }

            queue.write_buffer(
                &self.routed_garden_uniform_buffer,
                0,
                bytemuck::bytes_of(&HostRoutedGardenUniforms {
                    feedback_values: [
                        plan.uniforms.fb_zoom,
                        plan.uniforms.fb_rotate,
                        plan.uniforms.feedback_valid,
                        0.0,
                    ],
                    garden_values: plan.originals_uniforms.garden_values,
                    garden_modes: [
                        plan.originals_uniforms.garden_modes[2],
                        plan.originals_uniforms.garden_modes[3],
                        plan.originals_uniforms.garden_modes[0],
                        0,
                    ],
                }),
            );
            {
                let mut pass = begin_replace_pass(
                    encoder,
                    "Advanced composition routed Refresh Garden",
                    current_view,
                );
                pass.set_pipeline(&self.routed_garden_pipeline);
                pass.set_bind_group(0, &routed.bind_group, &[]);
                pass.set_bind_group(1, &self.routed_garden_uniform_group, &[]);
                pass.draw(0..3, 0..1);
            }
            render_copy(
                encoder,
                "Advanced composition commit routed Garden Compat8 feedback",
                &self.compat8_copy_pipeline,
                &input.current_copy_group,
                &self.temporal_feedback_view,
            );
            return true;
        }

        if !plan.legacy_shader_active && !plan.originals_shader_active {
            if let Some(history_write_target) = plan.history_write_target {
                render_copy(
                    encoder,
                    "Advanced composition record Compat8 clean history",
                    &self.compat8_copy_pipeline,
                    &input.current_copy_group,
                    &self.temporal_history_layer_views[history_write_target],
                );
            }
            // Frozen zero-mode behavior is a byte-preserving texture copy,
            // not a mathematically neutral shader pass.
            copy(encoder, current_texture, output_texture);
            render_copy(
                encoder,
                "Advanced composition zero-mode Compat8 feedback",
                &self.compat8_copy_pipeline,
                &input.current_copy_group,
                &self.temporal_feedback_view,
            );
            return false;
        }
        let mut advanced_uniforms = plan.uniforms;
        // The legacy ABI's final padding float remains zero in every shared
        // plan. Advanced alone claims value 1 at its private upload boundary
        // to select premultiplied temporal filtering/accumulation.
        advanced_uniforms._pad = 1.0;
        queue.write_buffer(
            &self.temporal_uniform_buffer,
            0,
            bytemuck::bytes_of(&advanced_uniforms),
        );
        let (pipeline, uniforms) = if plan.originals_shader_active {
            queue.write_buffer(
                &self.temporal_originals_uniform_buffer,
                0,
                bytemuck::bytes_of(&plan.originals_uniforms),
            );
            (
                &self.temporal_originals_pipeline,
                &self.temporal_originals_uniform_group,
            )
        } else {
            (&self.temporal_pipeline, &self.temporal_uniform_group)
        };
        {
            let mut pass =
                begin_replace_pass(encoder, "Advanced composition temporal", output_view);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &input.bind_group, &[]);
            pass.set_bind_group(1, uniforms, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(history_write_target) = plan.history_write_target {
            render_copy(
                encoder,
                "Advanced composition record Compat8 clean history",
                &self.compat8_copy_pipeline,
                &input.current_copy_group,
                &self.temporal_history_layer_views[history_write_target],
            );
        }
        render_copy(
            encoder,
            "Advanced composition commit Compat8 feedback",
            &self.compat8_copy_pipeline,
            &input.output_copy_group,
            &self.temporal_feedback_view,
        );
        false
    }

    #[allow(dead_code, reason = "compatibility wrapper for host tests/embedders")]
    pub fn reset_temporal(&self) {
        self.reset_temporal_for(TemporalResetCause::PatchGeneration);
    }

    pub fn reset_temporal_for(&self, cause: TemporalResetCause) {
        self.temporal_state.borrow_mut().reset_for(cause);
    }

    /// Clear authored temporal memory without broadening the reset to N-1
    /// Program history, image taps, or the exact audience image held by
    /// Program Freeze. The shared reset contract also rewinds Collision Score.
    pub fn clear_temporal_memory(&self) {
        self.temporal_state
            .borrow_mut()
            .reset_for(TemporalResetCause::ManualClear);
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "temporal validity inspection is exposed for host GPU goldens"
        )
    )]
    pub fn temporal_history_valid(&self) -> u32 {
        self.temporal_state.borrow().history_valid
    }

    /// The single D2Array **read** view of the committed Compat8 clean-history
    /// ring.
    ///
    /// This is a borrow, never a copy: a second consumer binds this same view
    /// and the ring stays exactly 24 `Rgba8UnormSrgb` layers charged once as
    /// `ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS`. Reading it alongside the
    /// temporal pass is safe because both are reads; the 24 single-layer views
    /// are render targets and are deliberately not exposed, because a second
    /// writer would corrupt the ring the temporal pass is mid-frame reading.
    pub fn temporal_history_view(&self) -> &wgpu::TextureView {
        &self.temporal_history_view
    }

    /// Which storage the 25-layer temporal class was built with. Production
    /// hosts always answer `Compat8`; only the measurement-only Gate 6
    /// constructor answers `Rgba16Float`.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "storage inspection is exposed for the Gate 6 measurement fixtures"
        )
    )]
    pub fn temporal_history_storage(&self) -> crate::precision::SurfaceStorage {
        self.temporal_history_storage
    }

    /// The clean-history read cursor a consumer must use to turn an age into a
    /// layer, derived through `temporal::temporal_read_snapshot` exactly as the
    /// temporal pass derives its own uniform.
    ///
    /// Age 0 is the virtual current image and never addresses a stored layer;
    /// an age is materialized only while it is strictly below `valid`.
    pub fn temporal_history_read_cursor(&self) -> (u32, u32) {
        let state = self.temporal_state.borrow();
        let snapshot =
            crate::temporal::temporal_read_snapshot(state.history_write, state.history_valid, 1.0);
        (
            u32::try_from(snapshot.virtual_write).unwrap_or(0),
            snapshot.virtual_valid,
        )
    }

    #[allow(
        dead_code,
        reason = "native telemetry reaches this through the active Advanced executor only"
    )]
    pub fn temporal_state_metrics(&self) -> TemporalStateMetrics {
        self.temporal_state.borrow().metrics()
    }

    pub fn commit_temporal_frame(&self) {
        self.temporal_state.borrow_mut().commit_staged();
    }

    pub fn discard_temporal_frame(&self) {
        self.temporal_state.borrow_mut().discard_staged();
    }

    pub fn encode_present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &HostTextureInputs,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), CompositionHostError> {
        if target_format != HOST_PRESENT_FORMAT {
            return Err(CompositionHostError::PresentFormat(target_format));
        }
        let mut pass = begin_replace_pass(encoder, "Advanced composition present", target);
        pass.set_pipeline(&self.present_pipeline);
        pass.set_bind_group(0, &source.bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    /// View of the global clean Program image from exactly N-1. A missing
    /// allocation means the immutable graph declared no previous-frame taps.
    #[allow(
        dead_code,
        reason = "single-view accessor remains a compatibility seam for host adapters"
    )]
    pub fn program_history_view(&self) -> Option<&wgpu::TextureView> {
        let read = self.program_history_read_index.get();
        self.program_history
            .as_ref()
            .map(|surfaces| &surfaces[read].view)
    }

    /// Both preallocated ProgramHistory views, in stable A/B order. Prepared
    /// consumers build both bind-group parities once and choose the committed
    /// read index during encode; no bind group is rebuilt on a frame swap.
    pub fn program_history_views(&self) -> Option<[&wgpu::TextureView; 2]> {
        self.program_history
            .as_ref()
            .map(|surfaces| [&surfaces[0].view, &surfaces[1].view])
    }

    pub fn program_history_read_index(&self) -> usize {
        self.program_history_read_index.get()
    }

    pub fn program_history_write_index(&self) -> usize {
        1 - self.program_history_read_index.get()
    }

    pub fn program_history_initialized(&self) -> bool {
        self.program_history_initialized.get()
    }

    /// Record the final copy only after every current/previous consumer has
    /// encoded. The CPU-visible validity bit changes separately in `commit`,
    /// so an encoder abandoned after a later error cannot publish history.
    pub fn encode_stage_program_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        completed_program: &wgpu::Texture,
    ) {
        let Some(history) = &self.program_history else {
            return;
        };
        let write = self.program_history_write_index();
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: completed_program,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &history[write].texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.dimensions[0],
                height: self.dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        self.program_history_staged.set(true);
    }

    pub fn commit_program_history(&self) {
        if self.program_history_staged.get() {
            self.program_history_read_index
                .set(self.program_history_write_index());
            self.program_history_initialized.set(true);
            self.program_history_staged.set(false);
        }
    }

    pub fn discard_program_history_stage(&self) {
        self.program_history_staged.set(false);
    }

    pub fn reset_program_history(&self) {
        self.program_history_read_index.set(0);
        self.program_history_initialized.set(false);
        self.program_history_staged.set(false);
    }

    /// Publish every CPU-visible N-1/temporal state transition only after the
    /// caller successfully submits the complete frame encoder.
    pub fn commit_frame_history(&self) {
        self.commit_program_history();
        self.commit_temporal_frame();
    }

    /// Roll back CPU cadence/readiness when any later encode/present step
    /// rejects and the frame encoder is abandoned.
    pub fn discard_frame_history(&self) {
        self.discard_program_history_stage();
        self.discard_temporal_frame();
    }
}

fn advanced_effect_pass_uniform(value: &EffectPassUniforms) -> EffectPassUniforms {
    let mut advanced = *value;
    if advanced.spatial.modes[3] != 0 {
        // Mode 1 is the frozen LegacyExact active-spatial textureSample path.
        // CompositionHost is Advanced-only and selects the straight-storage,
        // premultiplied-filtering implementation with mode 2.
        advanced.spatial.modes[3] = 2;
    }
    advanced
}

fn begin_replace_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    target: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    })
}

/// Independent CPU reference for the host's straight→premultiplied A/B
/// interpolation followed by Program-over. RGB may remain HDR until present.
#[cfg(test)]
pub(crate) fn host_bus_reference(
    a: [f32; 4],
    b: [f32; 4],
    program: [f32; 4],
    crossfade: f32,
) -> [f32; 4] {
    let premultiply = |value: [f32; 4]| {
        let alpha = finite_unit(value[3], 0.0);
        [
            finite(value[0], 0.0) * alpha,
            finite(value[1], 0.0) * alpha,
            finite(value[2], 0.0) * alpha,
            alpha,
        ]
    };
    let a = premultiply(a);
    let b = premultiply(b);
    let program = premultiply(program);
    let t = finite_unit(crossfade, 0.5);
    let ab = std::array::from_fn::<_, 4, _>(|channel| a[channel] * (1.0 - t) + b[channel] * t);
    let keep = 1.0 - program[3];
    let output: [f32; 4] = std::array::from_fn(|channel| program[channel] + ab[channel] * keep);
    if output[3] <= 1.0e-6 {
        [0.0; 4]
    } else {
        [
            output[0] / output[3],
            output[1] / output[3],
            output[2] / output[3],
            output[3].clamp(0.0, 1.0),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(test)]
pub(crate) struct HostGroupAdmissionReference {
    /// Post transform/rack/matte and deliberately pre opacity/solo/A-B.
    pub output_tap: [f32; 4],
    /// Value admitted to the selected lane after group opacity/solo/bypass.
    pub admitted: [f32; 4],
}

#[cfg(test)]
pub(crate) fn host_group_admission_reference(
    post_matte: [f32; 4],
    opacity: f32,
    admitted: bool,
) -> HostGroupAdmissionReference {
    let output_tap = [
        finite(post_matte[0], 0.0),
        finite(post_matte[1], 0.0),
        finite(post_matte[2], 0.0),
        finite_unit(post_matte[3], 0.0),
    ];
    let lane_alpha = if admitted {
        output_tap[3] * finite_unit(opacity, 1.0)
    } else {
        0.0
    };
    HostGroupAdmissionReference {
        output_tap,
        admitted: [output_tap[0], output_tap[1], output_tap[2], lane_alpha],
    }
}

#[cfg(test)]
fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
fn finite_unit(value: f32, fallback: f32) -> f32 {
    finite(value, fallback).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        composite_bus_reference, BusAssignment, BusSample, PremultipliedRgba,
    };
    use crate::effects::params::{
        CollisionAtlasParams, RefreshGardenGate, RefreshGardenParams, TemporalInterpolation,
        TemporalLoomParams, TemporalOriginalsParams, TemporalTopology,
    };
    use crate::temporal::{TemporalFrameEvents, TemporalFreezeState};

    fn resource_plan(
        dimensions: [u32; 2],
        rgba16_surface_layers: u32,
        compat8_surface_layers: u32,
    ) -> CreativeResourcePlan {
        let pixels = u64::from(dimensions[0]) * u64::from(dimensions[1]);
        CreativeResourcePlan {
            output_size: dimensions,
            full_frame_passes: 0,
            logical_texture_lookups_per_pixel: 0,
            texture_samples_per_pixel: 0,
            retained_surface_layers: rgba16_surface_layers + compat8_surface_layers,
            rgba16_surface_layers,
            compat8_surface_layers,
            creative_bytes: pixels
                * (u64::from(rgba16_surface_layers) * 8 + u64::from(compat8_surface_layers) * 4),
        }
    }

    fn capacities(dimensions: [u32; 2], retain_program_history: bool) -> HostCapacities {
        HostCapacities {
            effect_slots: 4,
            composite_slots: 4,
            matte_slots: 4,
            retain_program_history,
            resources: resource_plan(
                dimensions,
                8 + u32::from(retain_program_history)
                    * (1 + ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS),
                ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS,
            ),
        }
    }

    /// The Gate 6 candidate's plan: the same topology with the 25-layer
    /// temporal class charged at 8 bytes per pixel instead of 4. The
    /// candidate must declare the truth to `validate_host_resource_plan`,
    /// which computes the class width from the requested history storage.
    fn full16_capacities(dimensions: [u32; 2]) -> HostCapacities {
        let pixels = u64::from(dimensions[0]) * u64::from(dimensions[1]);
        let mut capacities = capacities(dimensions, false);
        capacities.resources.creative_bytes = pixels
            * (u64::from(capacities.resources.rgba16_surface_layers) * 8
                + u64::from(ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS) * 8);
        capacities
    }

    fn assert_near(actual: [f32; 4], expected: [f32; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{actual} != {expected}"
            );
        }
    }

    fn srgb_to_linear(value: u8) -> f32 {
        let encoded = value as f32 / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    fn half_to_f32(value: u16) -> f32 {
        let sign = u32::from(value & 0x8000) << 16;
        let exponent = (value >> 10) & 0x1f;
        let fraction = value & 0x03ff;
        let bits = if exponent == 0 {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = u32::from(fraction);
                let mut exponent = -14_i32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                fraction &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | (u32::from(fraction) << 13)
        } else {
            sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13)
        };
        f32::from_bits(bits)
    }

    fn positive_f32_to_half(value: f32) -> u16 {
        let value = value.clamp(0.0, 1.0);
        let mut low = 0_u16;
        let mut high = 0x3c00_u16;
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if half_to_f32(middle) < value {
                low = middle;
            } else {
                high = middle;
            }
        }
        if (half_to_f32(low) - value).abs() <= (half_to_f32(high) - value).abs() {
            low
        } else {
            high
        }
    }

    fn repeated_half_rgba(pixel: [u16; 4], count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 8);
        for _ in 0..count {
            for channel in pixel {
                bytes.extend_from_slice(&channel.to_le_bytes());
            }
        }
        bytes
    }

    fn create_uploaded_half_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: [u32; 2],
        label: &'static str,
        bytes: &[u8],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_WORKING_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * 8),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_compat8_target(
        device: &wgpu::Device,
        dimensions: [u32; 2],
        label: &'static str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_PRESENT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn copy_compat8_to_readback(
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        buffer: &wgpu::Buffer,
        offset: u64,
        dimensions: [u32; 2],
        padded_row: u32,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(dimensions[1]),
                },
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
    }

    fn map_readback(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        dimensions: [u32; 2],
        padded_row: u32,
        offset: u64,
    ) -> Vec<u8> {
        map_readback_rows(
            device,
            buffer,
            dimensions[1],
            dimensions[0] * 4,
            padded_row,
            offset,
        )
    }

    fn map_readback_rows(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        height: u32,
        row_bytes: u32,
        padded_row: u32,
        offset: u64,
    ) -> Vec<u8> {
        let slice = buffer.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("precision readback GPU wait");
        receive.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let mut bytes = Vec::with_capacity(row_bytes as usize * height as usize);
        for y in 0..height as usize {
            let start = offset as usize + y * padded_row as usize;
            bytes.extend_from_slice(&mapped[start..start + row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        bytes
    }

    fn linear_to_srgb_byte(value: f32) -> u8 {
        let linear = value.clamp(0.0, 1.0);
        let encoded = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round().clamp(0.0, 255.0) as u8
    }

    fn exact_fixture_bytes(samples: &[[f32; 4]]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for sample in samples {
            bytes.extend_from_slice(&[
                linear_to_srgb_byte(sample[0]),
                linear_to_srgb_byte(sample[1]),
                linear_to_srgb_byte(sample[2]),
                (sample[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
        bytes
    }

    fn advanced_fixture_bytes(samples: &[[f32; 4]]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for sample in samples {
            for channel in sample {
                bytes.extend_from_slice(&positive_f32_to_half(*channel).to_le_bytes());
            }
        }
        bytes
    }

    fn read_texture_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        dimensions: [u32; 2],
        bytes_per_pixel: u32,
        label: &'static str,
    ) -> Vec<u8> {
        let row_bytes = dimensions[0] * bytes_per_pixel;
        let padded_row = (row_bytes + 255) & !255;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: u64::from(padded_row) * u64::from(dimensions[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(dimensions[1]),
                },
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        map_readback_rows(device, &readback, dimensions[1], row_bytes, padded_row, 0)
    }

    fn exact_linear_samples(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(4)
            .map(|pixel| {
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                    pixel[3] as f32 / 255.0,
                ]
            })
            .collect()
    }

    fn advanced_linear_samples(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(8)
            .map(|pixel| {
                std::array::from_fn(|channel| {
                    let byte = channel * 2;
                    half_to_f32(u16::from_le_bytes([pixel[byte], pixel[byte + 1]]))
                })
            })
            .collect()
    }

    fn upload_texture_bytes(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        dimensions: [u32; 2],
        bytes_per_pixel: u32,
        bytes: &[u8],
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * bytes_per_pixel),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
    }

    struct CompatEffectFixture {
        pipeline: wgpu::RenderPipeline,
        texture_group: wgpu::BindGroup,
        uniform_group: wgpu::BindGroup,
    }

    fn prepare_compat_effect_fixture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        uniforms: &EffectPassUniforms,
    ) -> CompatEffectFixture {
        let (pipeline, texture_layout, uniform_layout, _vertex) =
            crate::renderer::state::build_effects_pipeline(device);
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("M6 Compat8 effects linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("M6 Compat8 effects nearest sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("M6 Compat8 effects texture group"),
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
            ],
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("M6 Compat8 effects uniform"),
            size: std::mem::size_of::<EffectPassUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("M6 Compat8 effects uniform group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        CompatEffectFixture {
            pipeline,
            texture_group,
            uniform_group,
        }
    }

    fn encode_compat_effect(
        fixture: &CompatEffectFixture,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let mut pass = begin_replace_pass(encoder, "M6 Compat8 production effects", target);
        pass.set_pipeline(&fixture.pipeline);
        pass.set_bind_group(0, &fixture.texture_group, &[]);
        pass.set_bind_group(1, &fixture.uniform_group, &[]);
        pass.draw(0..3, 0..1);
    }

    #[test]
    fn bus_cpu_law_matches_independent_premultiplied_reference() {
        let a = [1.2, 0.1, 0.3, 0.25];
        let b = [0.0, 0.9, 0.2, 0.75];
        let program = [0.4, 0.2, 1.4, 0.5];
        let crossfade = 0.37;
        let expected = composite_bus_reference(
            [
                BusSample {
                    bus: BusAssignment::A,
                    pixel: PremultipliedRgba::from_straight_linear([a[0], a[1], a[2]], a[3]),
                },
                BusSample {
                    bus: BusAssignment::B,
                    pixel: PremultipliedRgba::from_straight_linear([b[0], b[1], b[2]], b[3]),
                },
                BusSample {
                    bus: BusAssignment::Program,
                    pixel: PremultipliedRgba::from_straight_linear(
                        [program[0], program[1], program[2]],
                        program[3],
                    ),
                },
            ],
            crossfade,
        )
        .output
        .0;
        let actual = host_bus_reference(a, b, program, crossfade);
        assert_near(
            [
                actual[0] * actual[3],
                actual[1] * actual[3],
                actual[2] * actual[3],
                actual[3],
            ],
            expected,
        );
    }

    #[test]
    fn group_output_tap_precedes_opacity_and_admission() {
        let pixel = [0.8, 0.4, 0.2, 0.75];
        let admitted = host_group_admission_reference(pixel, 0.2, true);
        assert_eq!(admitted.output_tap, pixel);
        assert_eq!(admitted.admitted, [0.8, 0.4, 0.2, 0.15]);
        let solo_rejected = host_group_admission_reference(pixel, 1.0, false);
        assert_eq!(solo_rejected.output_tap, pixel);
        assert_eq!(solo_rejected.admitted[3], 0.0);
    }

    #[test]
    fn temporal_reference_clock_is_display_rate_independent() {
        for (fps, expected_records) in [(24, 24), (30, 30), (60, 30)] {
            let mut state = TemporalState::default();
            let mut records = 0;
            for _ in 0..fps {
                if matches!(
                    state
                        .stage_frame(
                            &TemporalParams::default(),
                            TemporalFrameInput::legacy(1.0 / fps as f32, true),
                            [1_920, 1_080],
                        )
                        .action,
                    TemporalFrameAction::Advance {
                        record_history: true
                    }
                ) {
                    records += 1;
                }
                state.commit_staged();
            }
            assert_eq!(records, expected_records, "{fps} fps");
        }
    }

    #[test]
    fn exact_and_advanced_temporal_plans_match_all_loom_modes_and_fixed_atlas_seeds() {
        let topologies = [
            TemporalTopology::Linear,
            TemporalTopology::Radial,
            TemporalTopology::Spiral,
            TemporalTopology::Contour,
            TemporalTopology::Folded,
            TemporalTopology::Kaleidoscopic,
        ];
        let interpolations = [TemporalInterpolation::Floor, TemporalInterpolation::Linear];
        let seeds = [0, 1, 0x6a09_e667, u32::MAX];

        for fps in [24_u32, 30, 60] {
            for topology in topologies {
                for interpolation in interpolations {
                    for seed in seeds {
                        let params = TemporalParams {
                            originals: TemporalOriginalsParams {
                                loom: TemporalLoomParams {
                                    amount: 1.0,
                                    topology,
                                    interpolation,
                                    depth: 0.875,
                                    phase: -0.25,
                                    scale: 1.75,
                                    angle: 37.0,
                                    folds: 9,
                                    quantization: 7,
                                },
                                atlas: CollisionAtlasParams {
                                    amount: 0.8,
                                    seed,
                                    territories: 13,
                                    collision: 0.45,
                                },
                                ..TemporalOriginalsParams::default()
                            },
                            ..TemporalParams::default()
                        };
                        let mut exact = TemporalState::default();
                        let mut advanced = TemporalState::default();
                        for frame in 0..(fps * 2) {
                            let input = TemporalFrameInput::legacy(1.0 / fps as f32, true);
                            let exact_plan = exact.stage_frame(&params, input, [13, 7]);
                            let advanced_plan = advanced.stage_frame(&params, input, [13, 7]);
                            assert_eq!(
                                advanced_plan, exact_plan,
                                "fps={fps}, frame={frame}, topology={topology:?}, interpolation={interpolation:?}, seed={seed}"
                            );
                            exact.commit_staged();
                            advanced.commit_staged();
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn exact_temporal_originals_keep_frozen_pixels_while_advanced_is_deterministic_at_24_30_and_60_fps(
    ) {
        use sha2::{Digest, Sha256};
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for temporal parity test");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Temporal Exact/Advanced parity device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for temporal parity test");
        let dimensions = [9_u32, 7_u32];
        let texture_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let exact_textures: [wgpu::Texture; 3] = std::array::from_fn(|index| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if index == 0 {
                    "Exact temporal parity clean/output"
                } else {
                    "Exact temporal parity scratch"
                }),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HOST_PRESENT_FORMAT,
                usage: texture_usage,
                view_formats: &[],
            })
        });
        let exact_views = exact_textures
            .each_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let (legacy_pipeline, texture_layout, legacy_uniform_layout) =
            crate::renderer::state::build_temporal_pipeline(&device);
        let (exact_history_texture, exact_history_view) =
            crate::renderer::state::build_history_texture(&device, dimensions[0], dimensions[1]);
        let (exact_feedback_texture, exact_feedback_view) =
            crate::renderer::state::build_feedback_texture(&device, dimensions[0], dimensions[1]);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Temporal Exact/Advanced parity sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let exact_resources = crate::renderer::state::build_prepared_temporal_gpu_resources(
            &device,
            &texture_layout,
            &legacy_uniform_layout,
            &exact_views[0],
            &exact_history_view,
            &sampler,
            &exact_feedback_view,
        );
        let mut exact_state = TemporalState::default();

        let advanced = CompositionHost::new(&device, dimensions, capacities(dimensions, false))
            .expect("advanced temporal parity host");
        let advanced_current = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Advanced temporal parity clean input"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_WORKING_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let advanced_current_view =
            advanced_current.create_view(&wgpu::TextureViewDescriptor::default());
        let advanced_output = advanced.surface(HostSurface::Pong);
        let advanced_input =
            advanced.prepare_temporal_input(&device, &advanced_current_view, advanced_output.view);
        let warmed = advanced.allocation_snapshot();

        let topologies = [
            TemporalTopology::Linear,
            TemporalTopology::Radial,
            TemporalTopology::Spiral,
            TemporalTopology::Contour,
            TemporalTopology::Folded,
            TemporalTopology::Kaleidoscopic,
        ];
        let interpolations = [TemporalInterpolation::Floor, TemporalInterpolation::Linear];
        let seeds = [0, 1, 0x6a09_e667, u32::MAX];
        let cases = 3_usize * topologies.len() * interpolations.len() * seeds.len();
        let padded_row = 256_u64;
        let surface_stride = padded_row * u64::from(dimensions[1]);
        let case_stride = surface_stride * 2;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temporal Exact/Advanced parity matrix readback"),
            size: case_stride * cases as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut diagnostics = Vec::with_capacity(cases);
        let mut case_index = 0_usize;

        for fps in [24_u32, 30, 60] {
            for topology in topologies {
                for interpolation in interpolations {
                    for seed in seeds {
                        let garden_gate = match (topology, interpolation) {
                            (TemporalTopology::Linear, TemporalInterpolation::Floor) => {
                                RefreshGardenGate::TemporalDelta
                            }
                            (TemporalTopology::Linear, TemporalInterpolation::Linear) => {
                                RefreshGardenGate::Matte
                            }
                            (TemporalTopology::Radial, _) => RefreshGardenGate::Luma,
                            (TemporalTopology::Spiral, _) => RefreshGardenGate::Chroma,
                            (TemporalTopology::Contour, _) => RefreshGardenGate::CellularRidge,
                            (TemporalTopology::Folded, _) => RefreshGardenGate::AudioEnergy,
                            (TemporalTopology::Kaleidoscopic, _) => RefreshGardenGate::AudioOnset,
                        };
                        let params = TemporalParams {
                            originals: TemporalOriginalsParams {
                                loom: TemporalLoomParams {
                                    amount: 1.0,
                                    topology,
                                    interpolation,
                                    depth: 0.9,
                                    phase: 0.17,
                                    scale: 1.4,
                                    angle: 31.0,
                                    folds: 7,
                                    quantization: 0,
                                },
                                atlas: CollisionAtlasParams {
                                    amount: 0.72,
                                    seed,
                                    territories: 12,
                                    collision: 0.55,
                                },
                                garden: RefreshGardenParams {
                                    amount: 0.37,
                                    gate: garden_gate,
                                    threshold: 0.3,
                                    softness: 0.05,
                                    decay: 0.97,
                                    max_hold_ticks: 7,
                                    ..RefreshGardenParams::default()
                                },
                                ..TemporalOriginalsParams::default()
                            },
                            ..TemporalParams::default()
                        };
                        exact_state.reset();
                        advanced.reset_temporal();
                        for frame in 0..15_u32 {
                            let mut exact_bytes = Vec::with_capacity(
                                dimensions[0] as usize * dimensions[1] as usize * 4,
                            );
                            let mut advanced_bytes = Vec::with_capacity(
                                dimensions[0] as usize * dimensions[1] as usize * 8,
                            );
                            for y in 0..dimensions[1] {
                                for x in 0..dimensions[0] {
                                    let bits = x
                                        .wrapping_mul(3)
                                        .wrapping_add(y.wrapping_mul(5))
                                        .wrapping_add(frame.wrapping_mul(7))
                                        .wrapping_add(seed);
                                    let rgba = [
                                        if bits & 1 == 0 { 0 } else { 255 },
                                        if bits & 2 == 0 { 0 } else { 255 },
                                        if bits & 4 == 0 { 0 } else { 255 },
                                        255,
                                    ];
                                    exact_bytes.extend_from_slice(&rgba);
                                    for channel in rgba {
                                        let half = if channel == 0 { 0_u16 } else { 0x3c00 };
                                        advanced_bytes.extend_from_slice(&half.to_le_bytes());
                                    }
                                }
                            }
                            queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &exact_textures[0],
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                &exact_bytes,
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
                            queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &advanced_current,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                &advanced_bytes,
                                wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(dimensions[0] * 8),
                                    rows_per_image: Some(dimensions[1]),
                                },
                                wgpu::Extent3d {
                                    width: dimensions[0],
                                    height: dimensions[1],
                                    depth_or_array_layers: 1,
                                },
                            );
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Temporal Exact/Advanced parity frame"),
                                });
                            let frame_input = TemporalFrameInput::new(
                                1.0 / fps as f32,
                                TemporalFreezeState::Running,
                                frame % 11 == 7,
                                TemporalFrameEvents {
                                    audio_onset_events: u32::from(frame % 5 == 0),
                                    ..TemporalFrameEvents::default()
                                },
                            )
                            .with_audio_energy(0.7);
                            crate::renderer::state::encode_temporal_prepared_frame(
                                &queue,
                                &mut encoder,
                                &params,
                                &legacy_pipeline,
                                &exact_resources,
                                &exact_textures,
                                &exact_views,
                                &exact_history_texture,
                                &exact_feedback_texture,
                                &mut exact_state,
                                frame_input,
                                dimensions[0],
                                dimensions[1],
                            );
                            advanced.encode_temporal(
                                &queue,
                                &mut encoder,
                                &advanced_input,
                                &advanced_current,
                                advanced_output.texture,
                                advanced_output.view,
                                &params,
                                HostFrameTiming::from_temporal_input(frame_input),
                            );
                            if frame == 14 {
                                let base = case_index as u64 * case_stride;
                                for (texture, offset, bytes_per_pixel) in [
                                    (&exact_textures[0], base, 4_u32),
                                    (advanced_output.texture, base + surface_stride, 8_u32),
                                ] {
                                    encoder.copy_texture_to_buffer(
                                        wgpu::TexelCopyTextureInfo {
                                            texture,
                                            mip_level: 0,
                                            origin: wgpu::Origin3d::ZERO,
                                            aspect: wgpu::TextureAspect::All,
                                        },
                                        wgpu::TexelCopyBufferInfo {
                                            buffer: &readback,
                                            layout: wgpu::TexelCopyBufferLayout {
                                                offset,
                                                bytes_per_row: Some(padded_row as u32),
                                                rows_per_image: Some(dimensions[1]),
                                            },
                                        },
                                        wgpu::Extent3d {
                                            width: dimensions[0],
                                            height: dimensions[1],
                                            depth_or_array_layers: 1,
                                        },
                                    );
                                    assert!(dimensions[0] * bytes_per_pixel <= padded_row as u32);
                                }
                            }
                            queue.submit(std::iter::once(encoder.finish()));
                            exact_state.commit_staged();
                            advanced.commit_temporal_frame();
                            assert_eq!(advanced.allocation_snapshot(), warmed);
                        }
                        diagnostics.push((fps, topology, interpolation, seed));
                        case_index += 1;
                    }
                }
            }
        }
        assert_eq!(case_index, cases);

        let slice = readback.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("temporal parity GPU wait");
        receive.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let mut exact_digest = Sha256::new();
        let mut advanced_digest = Sha256::new();
        for (index, (_fps, _topology, _interpolation, _seed)) in diagnostics.into_iter().enumerate()
        {
            let base = index * case_stride as usize;
            for y in 0..dimensions[1] as usize {
                let exact_row = base + y * padded_row as usize;
                let advanced_row = base + surface_stride as usize + y * padded_row as usize;
                for x in 0..dimensions[0] as usize {
                    let exact_pixel = &mapped[exact_row + x * 4..exact_row + x * 4 + 4];
                    let advanced_pixel = &mapped[advanced_row + x * 8..advanced_row + x * 8 + 8];
                    exact_digest.update(exact_pixel);
                    advanced_digest.update(advanced_pixel);
                }
            }
        }
        assert_eq!(
            format!("{:x}", exact_digest.finalize()),
            "27eac2e718096ca775866d507ee1fd8445e2429e18db79b36e5d8c163294525c",
            "Compat8 temporal Originals pixels changed"
        );
        assert_eq!(
            format!("{:x}", advanced_digest.finalize()),
            "b6b23cfdc9edd4a4d2653abc84327b85f407b06f69d1467bc745571c6bae1f81",
            "Advanced premultiplied temporal Originals pixels changed"
        );
        drop(mapped);
        readback.unmap();
    }

    #[test]
    fn temporal_reads_old_ring_for_key_and_virtual_slitscan() {
        let first = crate::temporal::temporal_read_snapshot(0, 0, 1.0);
        assert_eq!(first.virtual_write, 0);
        assert_eq!(first.virtual_valid, 0);
        assert_eq!(first.key_reference, None);

        // Depth one references the prior observation, never the current
        // clean image that will be recorded after the temporal pass.
        let second = crate::temporal::temporal_read_snapshot(0, 1, 1.0);
        assert_eq!(second.virtual_write, 1);
        assert_eq!(second.virtual_valid, 2);
        assert_eq!(second.key_reference, Some(0));

        // The virtual current slot is exposed even on a display frame where
        // the fixed 30-Hz history clock does not materialize a new copy.
        let no_record_display_frame = crate::temporal::temporal_read_snapshot(7, 8, 2.0);
        assert_eq!(no_record_display_frame.virtual_write, 8);
        assert_eq!(no_record_display_frame.virtual_valid, 9);
        assert_eq!(no_record_display_frame.key_reference, Some(6));

        let wrapped =
            crate::temporal::temporal_read_snapshot(HISTORY_LEN as usize - 1, HISTORY_LEN, 1.0);
        assert_eq!(wrapped.virtual_write, 0);
        assert_eq!(wrapped.virtual_valid, HISTORY_LEN);
        assert_eq!(wrapped.key_reference, Some(HISTORY_LEN as usize - 1));
    }

    #[test]
    fn resource_plan_is_rejected_before_gpu_allocation_when_underreported() {
        let dimensions = [1920, 1080];
        let mut underreported = capacities(dimensions, true);
        underreported.resources.compat8_surface_layers = 24;
        underreported.resources.retained_surface_layers -= 1;
        underreported.resources.creative_bytes -=
            u64::from(dimensions[0]) * u64::from(dimensions[1]) * 4;
        assert!(matches!(
            validate_host_resource_plan(
                dimensions,
                underreported,
                crate::precision::SurfaceStorage::Compat8
            ),
            Err(CompositionHostError::ResourcePlanUnderreports {
                planned_compat8: 24,
                required_compat8: ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS,
                ..
            })
        ));

        let admitted = capacities(dimensions, true);
        assert!(validate_host_resource_plan(
            dimensions,
            admitted,
            crate::precision::SurfaceStorage::Compat8
        )
        .is_ok());
        assert!(admitted.resources.creative_bytes < MAX_CREATIVE_GPU_BYTES);
    }

    #[test]
    fn host_shader_declares_premultiplied_bus_and_straight_output() {
        let shader = include_str!("../shaders/composition_host.wgsl");
        for law in [
            "fn premultiply",
            "let ab = mix",
            "premultiplied_over(program, ab)",
            "straight_from_premultiplied",
        ] {
            assert!(shader.contains(law), "missing host shader law: {law}");
        }
        for law in [
            "const BAYER_8X8",
            "fn ordered_compat8_dither",
            "fn fs_present",
            "srgb_to_linear(dithered)",
        ] {
            assert!(shader.contains(law), "missing Compat8 dither law: {law}");
        }
    }

    #[test]
    fn routed_garden_shader_is_post_temporal_premultiplied_and_binds_only_three_textures() {
        let routed = include_str!("../shaders/refresh_garden_routed.wgsl");
        assert!(routed.contains("@group(0) @binding(0) var current_tex"));
        assert!(routed.contains("@group(0) @binding(1) var feedback_tex"));
        assert!(routed.contains("@group(0) @binding(2) var signal_tex"));
        assert!(!routed.contains("@group(0) @binding(4)"));
        assert_eq!(routed.matches("textureLoad(feedback_tex").count(), 4);
        assert!(routed.contains("let current = premultiply(textureSampleLevel(current_tex"));
        assert!(routed.contains("let signal = select(routed.r, routed.a"));
        assert!(
            routed.contains("straight_from_premultiplied(carrier * retained + current * injected)")
        );

        let inline = include_str!("../shaders/temporal_originals.wgsl");
        assert_eq!(
            inline.matches("else if gate_mode == 6u").count(),
            2,
            "legacy and Advanced inline paths must both recognize Matte"
        );
        let (legacy_inline, advanced_inline) = inline
            .split_once("fn premultiply_originals")
            .expect("legacy and Advanced Originals source boundary");
        assert!(
            legacy_inline.contains("signal = current.a"),
            "LegacyExact must retain its frozen current-alpha Matte law"
        );
        assert!(
            !advanced_inline.contains("signal = current.a"),
            "Advanced Matte must remain closed until the routed pass supplies it"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn advanced_present_dithers_sub_lsb_and_filters_transparent_edges_premultiplied() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for Advanced dither proof");
        let float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
        assert!(
            adapter.features().contains(float32_filterable),
            "physical dither proof requires filterable Rgba32Float input so encoded 100.25 is not pre-quantized to half"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Advanced dither proof device"),
            required_features: float32_filterable,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for Advanced dither proof");
        let dimensions = [8_u32, 8_u32];
        let pixels = (dimensions[0] * dimensions[1]) as usize;
        let host = CompositionHost::new(&device, dimensions, capacities(dimensions, false))
            .expect("Advanced dither proof host");

        let encoded = 100.25_f32 / 255.0;
        let subtle_linear = if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        };
        let subtle_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Advanced exact-f32 sub-LSB source"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let subtle_samples = vec![[subtle_linear, subtle_linear, subtle_linear, 1.0]; pixels];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &subtle_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&subtle_samples),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * 16),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        let subtle_view = subtle_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let subtle_input = host.prepare_copy_source(&device, &subtle_view);
        let (target_a, target_a_view) =
            create_compat8_target(&device, dimensions, "Advanced dither target A");
        let (target_b, target_b_view) =
            create_compat8_target(&device, dimensions, "Advanced dither target B");
        let padded_row = 256_u32;
        let surface_bytes = u64::from(padded_row) * u64::from(dimensions[1]);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Advanced dither proof readback"),
            size: surface_bytes * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Advanced deterministic dither proof"),
        });
        host.encode_present(
            &mut encoder,
            &subtle_input,
            &target_a_view,
            HOST_PRESENT_FORMAT,
        )
        .unwrap();
        host.encode_present(
            &mut encoder,
            &subtle_input,
            &target_b_view,
            HOST_PRESENT_FORMAT,
        )
        .unwrap();
        copy_compat8_to_readback(
            &mut encoder,
            &target_a,
            &readback,
            0,
            dimensions,
            padded_row,
        );
        copy_compat8_to_readback(
            &mut encoder,
            &target_b,
            &readback,
            surface_bytes,
            dimensions,
            padded_row,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let both = map_readback(
            &device,
            &readback,
            [dimensions[0], dimensions[1] * 2],
            padded_row,
            0,
        );
        let first_len = pixels * 4;
        let (first, second) = both.split_at(first_len);
        assert_eq!(first, second, "static dither must be frame invariant");
        let channel_codes = first
            .chunks_exact(4)
            .map(|pixel| pixel[0])
            .collect::<Vec<_>>();
        assert_eq!(
            channel_codes.iter().filter(|code| **code == 100).count(),
            48,
            "quarter-LSB Bayer tile must retain code 100 in 48 cells"
        );
        assert_eq!(
            channel_codes.iter().filter(|code| **code == 101).count(),
            16,
            "quarter-LSB Bayer tile must promote exactly 16 cells"
        );
        assert_eq!(
            channel_codes
                .iter()
                .map(|code| u32::from(*code))
                .sum::<u32>(),
            64 * 100 + 16,
            "the tile mean must remain exactly code 100.25"
        );
        let mut levels = channel_codes;
        levels.sort_unstable();
        levels.dedup();
        assert_eq!(
            levels.len(),
            2,
            "a quarter-LSB signal must occupy two codes"
        );
        assert_eq!(
            levels[1],
            levels[0] + 1,
            "dither may span only adjacent codes"
        );
        assert!(first
            .chunks_exact(4)
            .all(|pixel| { pixel[0] == pixel[1] && pixel[1] == pixel[2] && pixel[3] == 255 }));

        // Scale adjacent opaque-red and transparent hidden-green texels across
        // the output. Hardware straight-alpha interpolation would create a
        // yellow fringe. The production Advanced Effects spatial sampler
        // filters four independently premultiplied texels instead.
        let mut edge_source = repeated_half_rgba([0x3c00, 0, 0, 0x3c00], 1);
        edge_source.extend_from_slice(&repeated_half_rgba([0, 0x3c00, 0, 0], 1));
        let (_edge_source_texture, edge_source_view) = create_uploaded_half_texture(
            &device,
            &queue,
            [2, 1],
            "Advanced adjacent transparent-edge source",
            &edge_source,
        );
        let edge_effect = host.prepare_effect_source(&device, &edge_source_view);
        let edge_working = host.surface(HostSurface::GroupScratch);
        host.write_effect_uniform(
            &queue,
            HostUniformSlot(0),
            &EffectPassUniforms::for_target(
                crate::effects::EffectUniforms::default(),
                crate::spatial::SpatialTransform {
                    edge: crate::spatial::EdgeMode::Clamp,
                    ..crate::spatial::SpatialTransform::default()
                },
                (2, 1),
                (dimensions[0], dimensions[1]),
            ),
        )
        .unwrap();
        let edge_present = host.prepare_copy_source(&device, edge_working.view);
        let (edge_target, edge_target_view) =
            create_compat8_target(&device, dimensions, "Advanced half-alpha edge target");
        let edge_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Advanced half-alpha edge readback"),
            size: surface_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut edge_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Advanced half-alpha edge proof"),
        });
        host.encode_effect(
            &mut edge_encoder,
            &edge_effect,
            edge_working.view,
            HostUniformSlot(0),
        )
        .unwrap();
        host.encode_present(
            &mut edge_encoder,
            &edge_present,
            &edge_target_view,
            HOST_PRESENT_FORMAT,
        )
        .unwrap();
        copy_compat8_to_readback(
            &mut edge_encoder,
            &edge_target,
            &edge_readback,
            0,
            dimensions,
            padded_row,
        );
        queue.submit(std::iter::once(edge_encoder.finish()));
        let edge = map_readback(&device, &edge_readback, dimensions, padded_row, 0);
        let mut partial_coverage = 0_usize;
        for pixel in edge.chunks_exact(4) {
            assert_eq!(pixel[1], 0, "hidden green fringed into {pixel:?}");
            assert_eq!(pixel[2], 0, "unexpected blue in {pixel:?}");
            if pixel[3] == 0 {
                assert_eq!(pixel[0], 0, "transparent RGB was not canonicalized");
            } else {
                assert!(
                    pixel[0] >= 254,
                    "covered edge lost red chroma after half/present rounding: {pixel:?}"
                );
            }
            partial_coverage += usize::from(pixel[3] > 0 && pixel[3] < 255);
        }
        assert!(
            partial_coverage >= 2,
            "fixture did not exercise bilinear coverage"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn advanced_temporal_feedback_filters_hidden_rgb_in_premultiplied_space() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for Advanced temporal edge proof");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Advanced temporal edge proof device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for Advanced temporal edge proof");
        let dimensions = [8_u32, 8_u32];
        let host = CompositionHost::new(&device, dimensions, capacities(dimensions, false))
            .expect("Advanced temporal edge host");

        let mut edge_bytes = Vec::with_capacity(8 * 8 * 8);
        for _y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let pixel: [u16; 4] = if x < dimensions[0] / 2 {
                    [0x3c00, 0, 0, 0x3c00]
                } else {
                    [0, 0x3c00, 0, 0]
                };
                for channel in pixel {
                    edge_bytes.extend_from_slice(&channel.to_le_bytes());
                }
            }
        }
        let (edge_texture, edge_view) = create_uploaded_half_texture(
            &device,
            &queue,
            dimensions,
            "Advanced temporal hidden-RGB edge",
            &edge_bytes,
        );
        let output = host.surface(HostSurface::Pong);
        let edge_input = host.prepare_temporal_input(&device, &edge_view, output.view);
        let timing =
            HostFrameTiming::from_temporal_input(TemporalFrameInput::legacy(1.0 / 30.0, true));
        let mut prime = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Advanced temporal hidden-RGB prime"),
        });
        host.encode_temporal(
            &queue,
            &mut prime,
            &edge_input,
            &edge_texture,
            output.texture,
            output.view,
            &TemporalParams::default(),
            timing,
        );
        queue.submit(std::iter::once(prime.finish()));
        host.commit_temporal_frame();

        let black = repeated_half_rgba([0, 0, 0, 0], 64);
        let (black_texture, black_view) = create_uploaded_half_texture(
            &device,
            &queue,
            dimensions,
            "Advanced temporal transparent current",
            &black,
        );
        let black_input = host.prepare_temporal_input(&device, &black_view, output.view);
        let params = TemporalParams {
            feedback: 0.98,
            fb_zoom: 1.35,
            ..TemporalParams::default()
        };
        let padded_row = 256_u32;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Advanced temporal hidden-RGB readback"),
            size: u64::from(padded_row) * u64::from(dimensions[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Advanced temporal hidden-RGB proof"),
        });
        host.encode_temporal(
            &queue,
            &mut encoder,
            &black_input,
            &black_texture,
            output.texture,
            output.view,
            &params,
            timing,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(dimensions[1]),
                },
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let bytes = map_readback_rows(
            &device,
            &readback,
            dimensions[1],
            dimensions[0] * 8,
            padded_row,
            0,
        );
        let mut partial_coverage = 0_usize;
        for pixel in bytes.chunks_exact(8) {
            let channels = std::array::from_fn::<_, 4, _>(|channel| {
                let byte = channel * 2;
                half_to_f32(u16::from_le_bytes([pixel[byte], pixel[byte + 1]]))
            });
            assert!(
                channels[1].abs() <= 1.0e-6,
                "hidden green fringe: {channels:?}"
            );
            assert!(channels[2].abs() <= 1.0e-6, "unexpected blue: {channels:?}");
            if channels[3] <= 1.0e-6 {
                assert!(channels[0].abs() <= 1.0e-6, "transparent RGB: {channels:?}");
            } else {
                assert!(channels[0] >= 0.999, "covered trail lost red: {channels:?}");
            }
            partial_coverage += usize::from(channels[3] > 1.0e-6 && channels[3] < 0.979);
        }
        assert!(
            partial_coverage >= 2,
            "zoomed feedback did not exercise partial coverage"
        );
    }

    #[test]
    #[ignore = "requires a physical GPU adapter; emits the M6 objective receipt"]
    fn gpu_precision_receipt_measures_real_still_and_temporal_workloads() {
        use sha2::{Digest, Sha256};
        use std::time::Instant;

        use crate::precision::{measure_precision, LinearRgbaFixture};

        const STILL_FRAMES: u32 = 8;
        const TEMPORAL_FRAMES: u32 = 12;
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("physical GPU adapter for M6 precision receipt");
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("M6 physical GPU precision receipt"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for M6 precision receipt");
        let dimensions = [192_u32, 108_u32];
        let texture_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let exact_textures: [wgpu::Texture; 3] = std::array::from_fn(|index| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if index == 0 {
                    "M6 Compat8 clean/output"
                } else {
                    "M6 Compat8 scratch"
                }),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HOST_PRESENT_FORMAT,
                usage: texture_usage,
                view_formats: &[],
            })
        });
        let exact_views = exact_textures
            .each_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let (exact_pipeline, exact_texture_layout, exact_uniform_layout) =
            crate::renderer::state::build_temporal_pipeline(&device);
        let (exact_history_texture, exact_history_view) =
            crate::renderer::state::build_history_texture(&device, dimensions[0], dimensions[1]);
        let (exact_feedback_texture, exact_feedback_view) =
            crate::renderer::state::build_feedback_texture(&device, dimensions[0], dimensions[1]);
        let exact_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("M6 Compat8 temporal sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let exact_resources = crate::renderer::state::build_prepared_temporal_gpu_resources(
            &device,
            &exact_texture_layout,
            &exact_uniform_layout,
            &exact_views[0],
            &exact_history_view,
            &exact_sampler,
            &exact_feedback_view,
        );
        let mut exact_state = TemporalState::default();

        let advanced = CompositionHost::new(&device, dimensions, capacities(dimensions, false))
            .expect("M6 Advanced composition host");
        let advanced_current = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("M6 Advanced clean input"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_WORKING_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let advanced_current_view =
            advanced_current.create_view(&wgpu::TextureViewDescriptor::default());
        let advanced_output = advanced.surface(HostSurface::Pong);
        let advanced_input =
            advanced.prepare_temporal_input(&device, &advanced_current_view, advanced_output.view);
        let advanced_still_working = advanced.surface(HostSurface::Ping);
        let advanced_still_effect = advanced.prepare_effect_source(&device, &advanced_current_view);
        let still_effects = crate::effects::EffectUniforms {
            brightness: 0.013,
            contrast: 0.11,
            ..crate::effects::EffectUniforms::default()
        };
        let still_uniforms = EffectPassUniforms::for_target(
            still_effects,
            crate::spatial::SpatialTransform::default(),
            (dimensions[0], dimensions[1]),
            (dimensions[0], dimensions[1]),
        );
        advanced
            .write_effect_uniform(&queue, HostUniformSlot(0), &still_uniforms)
            .unwrap();
        let exact_still_effect =
            prepare_compat_effect_fixture(&device, &queue, &exact_views[0], &still_uniforms);
        let advanced_still_present =
            advanced.prepare_copy_source(&device, advanced_still_working.view);
        let advanced_temporal_present = advanced.prepare_copy_source(&device, advanced_output.view);
        let advanced_present_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("M6 Advanced presented output"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_PRESENT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let advanced_present_view =
            advanced_present_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let frame_input = TemporalFrameInput::legacy(1.0 / 30.0, true);
        let timing = HostFrameTiming::from_temporal_input(frame_input);

        let still_samples = (0..dimensions[1])
            .flat_map(|y| {
                (0..dimensions[0]).map(move |x| {
                    let xn = x as f32 / (dimensions[0] - 1) as f32;
                    let yn = y as f32 / (dimensions[1] - 1) as f32;
                    let micro = ((x + y * 3) % 7) as f32 * 0.000_11;
                    [
                        0.006 + xn * 0.29 + yn * 0.013 + micro,
                        0.009 + xn * 0.21 + yn * 0.019 + micro * 0.7,
                        0.004 + xn * 0.13 + yn * 0.027 + micro * 1.3,
                        1.0,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let still_reference_samples = still_samples
            .iter()
            .map(|sample| {
                let factor = 1.0 + still_effects.contrast * 2.0;
                [
                    ((sample[0] + still_effects.brightness - 0.5) * factor + 0.5).clamp(0.0, 1.0),
                    ((sample[1] + still_effects.brightness - 0.5) * factor + 0.5).clamp(0.0, 1.0),
                    ((sample[2] + still_effects.brightness - 0.5) * factor + 0.5).clamp(0.0, 1.0),
                    sample[3],
                ]
            })
            .collect::<Vec<_>>();
        let still_exact_input = exact_fixture_bytes(&still_samples);
        let still_advanced_input = advanced_fixture_bytes(&still_samples);
        upload_texture_bytes(
            &queue,
            &exact_textures[0],
            dimensions,
            4,
            &still_exact_input,
        );
        upload_texture_bytes(
            &queue,
            &advanced_current,
            dimensions,
            8,
            &still_advanced_input,
        );

        let still_exact_start = Instant::now();
        for _ in 0..STILL_FRAMES {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M6 Compat8 still frame"),
            });
            encode_compat_effect(&exact_still_effect, &mut encoder, &exact_views[1]);
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Compat8 still wait");
        }
        let still_exact_ns = still_exact_start.elapsed().as_nanos() as u64;

        let still_advanced_start = Instant::now();
        for _ in 0..STILL_FRAMES {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M6 Advanced still frame"),
            });
            advanced
                .encode_effect(
                    &mut encoder,
                    &advanced_still_effect,
                    advanced_still_working.view,
                    HostUniformSlot(0),
                )
                .unwrap();
            advanced
                .encode_present(
                    &mut encoder,
                    &advanced_still_present,
                    &advanced_present_view,
                    HOST_PRESENT_FORMAT,
                )
                .unwrap();
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Advanced still wait");
        }
        let still_advanced_ns = still_advanced_start.elapsed().as_nanos() as u64;
        let still_exact_output = read_texture_bytes(
            &device,
            &queue,
            &exact_textures[1],
            dimensions,
            4,
            "M6 Compat8 still readback",
        );
        let still_advanced_working_output = read_texture_bytes(
            &device,
            &queue,
            advanced_still_working.texture,
            dimensions,
            8,
            "M6 Advanced still working readback",
        );
        let still_advanced_presented_output = read_texture_bytes(
            &device,
            &queue,
            &advanced_present_texture,
            dimensions,
            4,
            "M6 Advanced still presented readback",
        );

        exact_state.reset();
        advanced.reset_temporal();
        let temporal_params = TemporalParams {
            feedback: 0.50,
            ..TemporalParams::default()
        };
        let temporal_frame = |frame: u32| {
            (0..dimensions[1])
                .flat_map(|y| {
                    (0..dimensions[0]).map(move |x| {
                        let xn = x as f32 / (dimensions[0] - 1) as f32;
                        let yn = y as f32 / (dimensions[1] - 1) as f32;
                        let center = (frame * 17) % dimensions[0];
                        let distance = x.abs_diff(center);
                        let pulse = match distance {
                            0 => 0.48,
                            1 => 0.27,
                            2 => 0.09,
                            _ => 0.0,
                        };
                        let micro = ((x * 5 + y * 7 + frame * 3) % 11) as f32 * 0.000_09;
                        [
                            (0.008 + xn * 0.16 + yn * 0.011 + pulse + micro).min(0.95),
                            (0.011 + xn * 0.09 + yn * 0.017 + pulse * 0.42 + micro).min(0.95),
                            (0.006 + xn * 0.05 + yn * 0.023 + pulse * 0.18 + micro).min(0.95),
                            1.0,
                        ]
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut temporal_reference = Vec::new();
        let mut temporal_fixture_bytes = Vec::new();
        for frame in 0..TEMPORAL_FRAMES {
            let current = temporal_frame(frame);
            for sample in &current {
                for channel in sample {
                    temporal_fixture_bytes.extend_from_slice(&channel.to_le_bytes());
                }
            }
            temporal_reference = if frame == 0 {
                current
            } else {
                current
                    .into_iter()
                    .zip(temporal_reference)
                    .map(|(current, previous)| {
                        [
                            current[0].max(previous[0] * temporal_params.feedback),
                            current[1].max(previous[1] * temporal_params.feedback),
                            current[2].max(previous[2] * temporal_params.feedback),
                            current[3],
                        ]
                    })
                    .collect()
            };
        }

        let temporal_exact_start = Instant::now();
        for frame in 0..TEMPORAL_FRAMES {
            let bytes = exact_fixture_bytes(&temporal_frame(frame));
            upload_texture_bytes(&queue, &exact_textures[0], dimensions, 4, &bytes);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M6 Compat8 temporal frame"),
            });
            crate::renderer::state::encode_temporal_prepared_frame(
                &queue,
                &mut encoder,
                &temporal_params,
                &exact_pipeline,
                &exact_resources,
                &exact_textures,
                &exact_views,
                &exact_history_texture,
                &exact_feedback_texture,
                &mut exact_state,
                frame_input,
                dimensions[0],
                dimensions[1],
            );
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Compat8 temporal wait");
            exact_state.commit_staged();
        }
        let temporal_exact_ns = temporal_exact_start.elapsed().as_nanos() as u64;

        let temporal_advanced_start = Instant::now();
        for frame in 0..TEMPORAL_FRAMES {
            let bytes = advanced_fixture_bytes(&temporal_frame(frame));
            upload_texture_bytes(&queue, &advanced_current, dimensions, 8, &bytes);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M6 Advanced temporal frame"),
            });
            advanced.encode_temporal(
                &queue,
                &mut encoder,
                &advanced_input,
                &advanced_current,
                advanced_output.texture,
                advanced_output.view,
                &temporal_params,
                timing,
            );
            advanced
                .encode_present(
                    &mut encoder,
                    &advanced_temporal_present,
                    &advanced_present_view,
                    HOST_PRESENT_FORMAT,
                )
                .unwrap();
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Advanced temporal wait");
            advanced.commit_temporal_frame();
        }
        let temporal_advanced_ns = temporal_advanced_start.elapsed().as_nanos() as u64;
        let temporal_exact_output = read_texture_bytes(
            &device,
            &queue,
            &exact_textures[0],
            dimensions,
            4,
            "M6 Compat8 temporal readback",
        );
        let temporal_advanced_working_output = read_texture_bytes(
            &device,
            &queue,
            advanced_output.texture,
            dimensions,
            8,
            "M6 Advanced temporal working readback",
        );
        let temporal_advanced_presented_output = read_texture_bytes(
            &device,
            &queue,
            &advanced_present_texture,
            dimensions,
            4,
            "M6 Advanced temporal presented readback",
        );

        let still_reference = LinearRgbaFixture::try_from_samples(still_reference_samples).unwrap();
        let still_exact =
            LinearRgbaFixture::try_from_samples(exact_linear_samples(&still_exact_output)).unwrap();
        let still_advanced_working = LinearRgbaFixture::try_from_samples(advanced_linear_samples(
            &still_advanced_working_output,
        ))
        .unwrap();
        let still_advanced_presented = LinearRgbaFixture::try_from_samples(exact_linear_samples(
            &still_advanced_presented_output,
        ))
        .unwrap();
        let temporal_reference_fixture =
            LinearRgbaFixture::try_from_samples(temporal_reference).unwrap();
        let temporal_exact =
            LinearRgbaFixture::try_from_samples(exact_linear_samples(&temporal_exact_output))
                .unwrap();
        let temporal_advanced_working = LinearRgbaFixture::try_from_samples(
            advanced_linear_samples(&temporal_advanced_working_output),
        )
        .unwrap();
        let temporal_advanced_presented = LinearRgbaFixture::try_from_samples(
            exact_linear_samples(&temporal_advanced_presented_output),
        )
        .unwrap();
        let still_exact_metrics = measure_precision(&still_reference, &still_exact).unwrap();
        let still_advanced_working_metrics =
            measure_precision(&still_reference, &still_advanced_working).unwrap();
        let still_advanced_presented_metrics =
            measure_precision(&still_reference, &still_advanced_presented).unwrap();
        let temporal_exact_metrics =
            measure_precision(&temporal_reference_fixture, &temporal_exact).unwrap();
        let temporal_advanced_working_metrics =
            measure_precision(&temporal_reference_fixture, &temporal_advanced_working).unwrap();
        let temporal_advanced_presented_metrics =
            measure_precision(&temporal_reference_fixture, &temporal_advanced_presented).unwrap();
        let block_mean_fixture = |fixture: &LinearRgbaFixture| {
            const BLOCK: usize = 8;
            let width = dimensions[0] as usize;
            let height = dimensions[1] as usize;
            let mut means = Vec::with_capacity((width / BLOCK) * (height / BLOCK));
            for block_y in 0..height / BLOCK {
                for block_x in 0..width / BLOCK {
                    let mut total = [0.0_f32; 4];
                    for y in block_y * BLOCK..(block_y + 1) * BLOCK {
                        for x in block_x * BLOCK..(block_x + 1) * BLOCK {
                            let sample = fixture.samples()[y * width + x];
                            for channel in 0..4 {
                                total[channel] += sample[channel];
                            }
                        }
                    }
                    means.push(total.map(|value| value / (BLOCK * BLOCK) as f32));
                }
            }
            LinearRgbaFixture::try_from_samples(means).unwrap()
        };
        let still_block_reference = block_mean_fixture(&still_reference);
        let still_block_exact = block_mean_fixture(&still_exact);
        let still_block_advanced_presented = block_mean_fixture(&still_advanced_presented);
        let temporal_block_reference = block_mean_fixture(&temporal_reference_fixture);
        let temporal_block_exact = block_mean_fixture(&temporal_exact);
        let temporal_block_advanced_presented = block_mean_fixture(&temporal_advanced_presented);
        let still_exact_block_metrics =
            measure_precision(&still_block_reference, &still_block_exact).unwrap();
        let still_advanced_presented_block_metrics =
            measure_precision(&still_block_reference, &still_block_advanced_presented).unwrap();
        let temporal_exact_block_metrics =
            measure_precision(&temporal_block_reference, &temporal_block_exact).unwrap();
        let temporal_advanced_presented_block_metrics = measure_precision(
            &temporal_block_reference,
            &temporal_block_advanced_presented,
        )
        .unwrap();
        assert!(
            still_advanced_working_metrics.rmse < still_exact_metrics.rmse,
            "Advanced still working path did not reduce objective error"
        );
        assert!(
            temporal_advanced_working_metrics.rmse < temporal_exact_metrics.rmse,
            "Advanced temporal working path did not reduce objective error: Compat8={temporal_exact_metrics:?}, Advanced={temporal_advanced_working_metrics:?}"
        );
        assert!(
            still_advanced_working_metrics.retained_gradient_events
                >= still_exact_metrics.retained_gradient_events
        );
        assert!(
            temporal_advanced_working_metrics.retained_gradient_events
                >= temporal_exact_metrics.retained_gradient_events
        );
        assert!(
            still_advanced_presented_block_metrics.rmse < still_exact_block_metrics.rmse,
            "Advanced dithered still presentation did not reduce 8x8 spatial-mean error"
        );
        assert!(
            temporal_advanced_presented_block_metrics.rmse < temporal_exact_block_metrics.rmse,
            "Advanced dithered temporal presentation did not reduce 8x8 spatial-mean error"
        );

        let mut still_fixture_bytes = Vec::with_capacity(still_samples.len() * 16);
        for sample in &still_samples {
            for channel in sample {
                still_fixture_bytes.extend_from_slice(&channel.to_le_bytes());
            }
        }
        let sha256 = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let digest_parts = |parts: &[&[u8]]| {
            let mut digest = Sha256::new();
            for part in parts {
                digest.update((part.len() as u64).to_le_bytes());
                digest.update(part);
            }
            format!("{:x}", digest.finalize())
        };
        let shader_bundle_sha256 = digest_parts(&[
            include_bytes!("../shaders/fullscreen.wgsl"),
            include_bytes!("../shaders/effects.wgsl"),
            include_bytes!("../shaders/temporal.wgsl"),
            include_bytes!("../shaders/composition_host.wgsl"),
        ]);
        let source_manifest_sha256 = digest_parts(&[
            include_bytes!("../../Cargo.toml"),
            include_bytes!("../../Cargo.lock"),
            include_bytes!("composition_host.rs"),
            include_bytes!("composition.rs"),
            include_bytes!("rack.rs"),
            include_bytes!("state.rs"),
            include_bytes!("motion.rs"),
            include_bytes!("blend.rs"),
            include_bytes!("compositor.rs"),
            include_bytes!("../temporal.rs"),
            include_bytes!("../precision.rs"),
            include_bytes!("../spatial.rs"),
            include_bytes!("../effects/mod.rs"),
            include_bytes!("../effects/params.rs"),
            include_bytes!("../visual_rack.rs"),
            include_bytes!("../evaluated_composition.rs"),
            include_bytes!("../render_export.rs"),
            include_bytes!("../shaders/fullscreen.wgsl"),
            include_bytes!("../shaders/effects.wgsl"),
            include_bytes!("../shaders/temporal.wgsl"),
            include_bytes!("../shaders/temporal_originals.wgsl"),
            include_bytes!("../shaders/refresh_garden_routed.wgsl"),
            include_bytes!("../shaders/composition_host.wgsl"),
            include_bytes!("../shaders/rack_node.wgsl"),
            include_bytes!("../shaders/motion_apply.wgsl"),
            include_bytes!("../shaders/motion_refresh.wgsl"),
            include_bytes!("../shaders/motion_luma.wgsl"),
            include_bytes!("../shaders/motion_lattice.wgsl"),
            include_bytes!("../shaders/motion_garden_signal.wgsl"),
        ]);
        let still_fixture_sha256 =
            digest_parts(&[&still_fixture_bytes, bytemuck::bytes_of(&still_uniforms)]);
        let temporal_fixture_sha256 = digest_parts(&[
            &temporal_fixture_bytes,
            &temporal_params.feedback.to_le_bytes(),
        ]);
        let still_compat8_sha256 = sha256(&still_exact_output);
        let still_advanced_working_sha256 = sha256(&still_advanced_working_output);
        let still_advanced_presented_sha256 = sha256(&still_advanced_presented_output);
        let temporal_compat8_sha256 = sha256(&temporal_exact_output);
        let temporal_advanced_working_sha256 = sha256(&temporal_advanced_working_output);
        let temporal_advanced_presented_sha256 = sha256(&temporal_advanced_presented_output);
        assert_eq!(
            shader_bundle_sha256,
            "1c30e098763ff6b1a89c56b89ca0af4fc98d489bebd03a41e5870bd940dab29c"
        );
        assert_eq!(
            still_fixture_sha256,
            "ddaa723b65d5d664ccb326e7bbe3acd3943391718a830e4f12b374a23ce32916"
        );
        assert_eq!(
            temporal_fixture_sha256,
            "b019456a5b64d4616f8cbd8027313d0a75f2fa1692937f81373f7deca7ed9b61"
        );
        assert_eq!(
            still_compat8_sha256,
            "5e5f663ddc8f72b0be7298960c22c7b5dd5d8eb5b0d8638abe1176dc253a855f"
        );
        assert_eq!(
            still_advanced_working_sha256,
            "25e6f3465943edd1eb5b952ccde80bc0813c01f4352bdf49a51ddaeea292cdad"
        );
        assert_eq!(
            still_advanced_presented_sha256,
            "b63f2077e59a745e740fff673b6dc4b56e0e4d05860ace9fad0e5123b2ab62cc"
        );
        assert_eq!(
            temporal_compat8_sha256,
            "e697af4694a76734584cc6c281c0d83d59893012291ec5611fcf619c1a9fb8c9"
        );
        assert_eq!(
            temporal_advanced_working_sha256,
            "644d4afa3b88284316b52c2ebfb19b82468170bcec4ab01abaa92babd4308893"
        );
        assert_eq!(
            temporal_advanced_presented_sha256,
            "e0bd88d853f78a86a06307daf94789928e2d3c167445275d2ee30dc00dc0b1ec"
        );
        let metric_json = |measurement: crate::precision::PrecisionMeasurement| {
            serde_json::json!({
                "rmse": measurement.rmse,
                "max_absolute_error": measurement.max_absolute_error,
                "clamped_channel_events": measurement.clamped_channel_events,
                "reference_gradient_events": measurement.reference_gradient_events,
                "retained_gradient_events": measurement.retained_gradient_events,
            })
        };
        let tool_version = |program: &str| {
            std::process::Command::new(program)
                .arg("--version")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .unwrap_or_else(|| "unavailable".to_string())
        };
        let receipt = serde_json::json!({
            "schema": "collide-o-scope-m6-gpu-precision-receipt/2",
            "command": "cargo test --locked gpu_precision_receipt_measures_real_still_and_temporal_workloads -- --ignored --nocapture",
            "host": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
            },
            "toolchain": {
                "rustc": tool_version("rustc"),
                "cargo": tool_version("cargo"),
            },
            "source_identity": {
                "crate_version": env!("CARGO_PKG_VERSION"),
                "shader_bundle_sha256": shader_bundle_sha256,
                "cargo_lock_and_m6_sources_sha256": source_manifest_sha256,
            },
            "timing_scope": "fixture-local wall time with a per-frame GPU wait; Compat8 writes its final target in the core pass while Advanced includes its required encode_present pass; smoke evidence only, not a renderer-throughput comparison",
            "adapter": {
                "name": adapter_info.name,
                "backend": format!("{:?}", adapter_info.backend),
                "device_type": format!("{:?}", adapter_info.device_type),
                "driver": adapter_info.driver,
                "driver_info": adapter_info.driver_info,
            },
            "dimensions": dimensions,
            "still": {
                "frames": STILL_FRAMES,
                "workload": "production effects shader: linear brightness + contrast, then Advanced deterministic Compat8 presentation",
                "fixture_sha256": still_fixture_sha256,
                "compat8_output_sha256": still_compat8_sha256,
                "advanced_working_output_sha256": still_advanced_working_sha256,
                "advanced_presented_output_sha256": still_advanced_presented_sha256,
                "compat8": metric_json(still_exact_metrics),
                "advanced_working_rgba16f": metric_json(still_advanced_working_metrics),
                "advanced_presented_compat8": metric_json(still_advanced_presented_metrics),
                "compat8_spatial_8x8_mean": metric_json(still_exact_block_metrics),
                "advanced_presented_spatial_8x8_mean": metric_json(still_advanced_presented_block_metrics),
                "compat8_fixture_wall_ns_per_frame": still_exact_ns / u64::from(STILL_FRAMES),
                "advanced_fixture_wall_ns_per_frame": still_advanced_ns / u64::from(STILL_FRAMES),
            },
            "temporal": {
                "frames": TEMPORAL_FRAMES,
                "feedback": temporal_params.feedback,
                "workload": "production feedback shader over a moving pulse, then Advanced deterministic Compat8 presentation",
                "fixture_sha256": temporal_fixture_sha256,
                "compat8_output_sha256": temporal_compat8_sha256,
                "advanced_working_output_sha256": temporal_advanced_working_sha256,
                "advanced_presented_output_sha256": temporal_advanced_presented_sha256,
                "compat8": metric_json(temporal_exact_metrics),
                "advanced_working_rgba16f": metric_json(temporal_advanced_working_metrics),
                "advanced_presented_compat8": metric_json(temporal_advanced_presented_metrics),
                "compat8_spatial_8x8_mean": metric_json(temporal_exact_block_metrics),
                "advanced_presented_spatial_8x8_mean": metric_json(temporal_advanced_presented_block_metrics),
                "compat8_fixture_wall_ns_per_frame": temporal_exact_ns / u64::from(TEMPORAL_FRAMES),
                "advanced_fixture_wall_ns_per_frame": temporal_advanced_ns / u64::from(TEMPORAL_FRAMES),
            },
        });
        println!(
            "M6_GPU_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }

    #[test]
    fn full16_history_plan_charges_eight_bytes_per_temporal_pixel_and_discriminates() {
        use crate::precision::SurfaceStorage;

        let dimensions = [64_u32, 36];
        let pixels = u64::from(dimensions[0]) * u64::from(dimensions[1]);

        // The settled plan validates under Compat8 exactly as it always has,
        // and is refused under the candidate storage — the widths genuinely
        // discriminate rather than both passing a looser check.
        assert!(validate_host_resource_plan(
            dimensions,
            capacities(dimensions, false),
            SurfaceStorage::Compat8,
        )
        .is_ok());
        assert!(matches!(
            validate_host_resource_plan(
                dimensions,
                capacities(dimensions, false),
                SurfaceStorage::Rgba16Float,
            ),
            Err(CompositionHostError::ResourcePlanBytes { .. })
        ));

        // The candidate plan validates only under the candidate storage.
        assert!(validate_host_resource_plan(
            dimensions,
            full16_capacities(dimensions),
            SurfaceStorage::Rgba16Float,
        )
        .is_ok());
        assert!(matches!(
            validate_host_resource_plan(
                dimensions,
                full16_capacities(dimensions),
                SurfaceStorage::Compat8,
            ),
            Err(CompositionHostError::ResourcePlanBytes { .. })
        ));

        // One byte under the exact total is refused, the ledger's own law.
        let mut under = full16_capacities(dimensions);
        under.resources.creative_bytes -= 1;
        assert!(matches!(
            validate_host_resource_plan(dimensions, under, SurfaceStorage::Rgba16Float),
            Err(CompositionHostError::ResourcePlanBytes { .. })
        ));

        // The exact increase is the documented candidate delta: 25 temporal
        // surfaces widening from 4 to 8 bytes per pixel.
        let settled = capacities(dimensions, false).resources.creative_bytes;
        let candidate = full16_capacities(dimensions).resources.creative_bytes;
        assert_eq!(
            candidate - settled,
            pixels * u64::from(ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS) * 4
        );
    }

    /// Gate 6's measurement, through the real composition host on the
    /// production device request: the `ExperimentalFull16History` candidate
    /// against the settled path on two temporal lanes, each judged against an
    /// analytic f32 reference no candidate output ever touches, with the
    /// verdict written into the tracked receipt
    /// `docs/evidence/full16-history-candidate-receipt.json` (regenerated in
    /// place — a changed receipt after an opt-in run is a new measurement on
    /// new hardware, not drift; commit it). The candidate is measurement-only:
    /// no production call site constructs it and the settled default does not
    /// move, which the M6 receipt's pinned output SHAs prove across this
    /// tranche.
    #[test]
    #[ignore = "requires a physical GPU adapter; emits the Gate 6 Full-16 history candidate receipt"]
    fn gpu_full16_history_candidate_measures_temporal_gain_and_writes_the_receipt() {
        use crate::precision::{
            measure_precision, ArtisticGainAssessment, LinearRgbaFixture, ObjectiveGainVerdict,
            PrecisionResourceDelta, SurfaceStorage,
        };

        const TEMPORAL_FRAMES: u32 = 12;
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("physical GPU adapter for the Full-16 candidate receipt");
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Gate 6 Full-16 history candidate receipt"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for the Full-16 candidate receipt");
        let dimensions = [192_u32, 108];
        let pixels = u64::from(dimensions[0]) * u64::from(dimensions[1]);

        let settled = CompositionHost::new(&device, dimensions, capacities(dimensions, false))
            .expect("settled host");
        let candidate = CompositionHost::new_with_history_storage(
            &device,
            dimensions,
            full16_capacities(dimensions),
            SurfaceStorage::Rgba16Float,
        )
        .expect("Full-16 candidate host");
        assert_eq!(settled.temporal_history_storage(), SurfaceStorage::Compat8);
        assert_eq!(
            candidate.temporal_history_storage(),
            SurfaceStorage::Rgba16Float
        );

        // The M6 temporal workload, verbatim: a moving pulse over a gradient,
        // feedback 0.50, and the analytic max/decay reference.
        let temporal_params = TemporalParams {
            feedback: 0.50,
            ..TemporalParams::default()
        };
        let temporal_frame = |frame: u32| {
            (0..dimensions[1])
                .flat_map(|y| {
                    (0..dimensions[0]).map(move |x| {
                        let xn = x as f32 / (dimensions[0] - 1) as f32;
                        let yn = y as f32 / (dimensions[1] - 1) as f32;
                        let center = (frame * 17) % dimensions[0];
                        let distance = x.abs_diff(center);
                        let pulse = match distance {
                            0 => 0.48,
                            1 => 0.27,
                            2 => 0.09,
                            _ => 0.0,
                        };
                        let micro = ((x * 5 + y * 7 + frame * 3) % 11) as f32 * 0.000_09;
                        [
                            (0.008 + xn * 0.16 + yn * 0.011 + pulse + micro).min(0.95),
                            (0.011 + xn * 0.09 + yn * 0.017 + pulse * 0.42 + micro).min(0.95),
                            (0.006 + xn * 0.05 + yn * 0.023 + pulse * 0.18 + micro).min(0.95),
                            1.0,
                        ]
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut feedback_reference = Vec::new();
        for frame in 0..TEMPORAL_FRAMES {
            let current = temporal_frame(frame);
            feedback_reference = if frame == 0 {
                current
            } else {
                current
                    .into_iter()
                    .zip(feedback_reference)
                    .map(|(current, previous)| {
                        [
                            current[0].max(previous[0] * temporal_params.feedback),
                            current[1].max(previous[1] * temporal_params.feedback),
                            current[2].max(previous[2] * temporal_params.feedback),
                            current[3],
                        ]
                    })
                    .collect()
            };
        }
        let feedback_reference_fixture =
            LinearRgbaFixture::try_from_samples(feedback_reference).unwrap();
        let frame_input = TemporalFrameInput::legacy(1.0 / 30.0, true);
        let timing = HostFrameTiming::from_temporal_input(frame_input);

        // One f16 readback target serves every lane on both hosts.
        let readback_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gate 6 candidate readback target"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_WORKING_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback_view = readback_target.create_view(&wgpu::TextureViewDescriptor::default());

        let run_lanes = |host: &CompositionHost| {
            let current = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Gate 6 lane clean input"),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HOST_WORKING_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let current_view = current.create_view(&wgpu::TextureViewDescriptor::default());
            let output = host.surface(HostSurface::Pong);
            let input = host.prepare_temporal_input(&device, &current_view, output.view);

            // Lane 1 — feedback recursion. The recursive member of the
            // temporal class re-quantizes every frame on the settled path.
            host.reset_temporal();
            for frame in 0..TEMPORAL_FRAMES {
                let bytes = advanced_fixture_bytes(&temporal_frame(frame));
                upload_texture_bytes(&queue, &current, dimensions, 8, &bytes);
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gate 6 feedback lane frame"),
                });
                host.encode_temporal(
                    &queue,
                    &mut encoder,
                    &input,
                    &current,
                    output.texture,
                    output.view,
                    &temporal_params,
                    timing,
                );
                queue.submit(std::iter::once(encoder.finish()));
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("feedback lane wait");
                host.commit_temporal_frame();
            }
            let feedback_output = read_texture_bytes(
                &device,
                &queue,
                output.texture,
                dimensions,
                8,
                "Gate 6 feedback lane readback",
            );
            let feedback_metrics = measure_precision(
                &feedback_reference_fixture,
                &LinearRgbaFixture::try_from_samples(advanced_linear_samples(&feedback_output))
                    .unwrap(),
            )
            .unwrap();

            // Lane 2 — clean-history storage fidelity: what one committed
            // ring layer returns against the exact frame it recorded, through
            // the production conversion pipeline in both directions.
            let ring_frame = temporal_frame(0);
            upload_texture_bytes(
                &queue,
                &current,
                dimensions,
                8,
                &advanced_fixture_bytes(&ring_frame),
            );
            let ring_layer_view =
                host._temporal_history_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("Gate 6 ring lane layer 0"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
            let ring_write_source = host.prepare_copy_source(&device, &current_view);
            let ring_read_source = host.prepare_copy_source(&device, &ring_layer_view);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gate 6 ring lane"),
            });
            {
                let mut pass =
                    begin_replace_pass(&mut encoder, "Gate 6 ring record", &ring_layer_view);
                pass.set_pipeline(&host.compat8_copy_pipeline);
                pass.set_bind_group(0, &ring_write_source.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let mut pass =
                    begin_replace_pass(&mut encoder, "Gate 6 ring readout", &readback_view);
                pass.set_pipeline(&host.copy_pipeline);
                pass.set_bind_group(0, &ring_read_source.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("ring lane wait");
            let ring_output = read_texture_bytes(
                &device,
                &queue,
                &readback_target,
                dimensions,
                8,
                "Gate 6 ring lane readback",
            );
            let ring_metrics = measure_precision(
                &LinearRgbaFixture::try_from_samples(ring_frame).unwrap(),
                &LinearRgbaFixture::try_from_samples(advanced_linear_samples(&ring_output))
                    .unwrap(),
            )
            .unwrap();
            (feedback_metrics, ring_metrics)
        };

        let (settled_feedback, settled_ring) = run_lanes(&settled);
        let (candidate_feedback, candidate_ring) = run_lanes(&candidate);

        // The exact candidate cost at this fixture size: 25 temporal
        // surfaces widening from 4 to 8 bytes per pixel.
        let delta = PrecisionResourceDelta {
            additional_bytes: pixels * u64::from(ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS) * 4,
            released_bytes: 0,
        };
        let feedback_assessment =
            ArtisticGainAssessment::compare(settled_feedback, candidate_feedback, delta).unwrap();
        let ring_assessment =
            ArtisticGainAssessment::compare(settled_ring, candidate_ring, delta).unwrap();

        // Storage fidelity must improve — f16 is denser than sRGB8 across
        // the whole in-gamut range — and the recursive lane must not regress.
        assert!(
            candidate_ring.rmse < settled_ring.rmse,
            "Full-16 ring storage did not reduce objective error: settled={settled_ring:?}, candidate={candidate_ring:?}"
        );
        assert!(
            candidate_feedback.rmse <= settled_feedback.rmse,
            "Full-16 feedback recursion regressed: settled={settled_feedback:?}, candidate={candidate_feedback:?}"
        );

        let verdict_str = |verdict: ObjectiveGainVerdict| match verdict {
            ObjectiveGainVerdict::NoMeasuredGain => "no measured gain",
            ObjectiveGainVerdict::MeasuredObjectiveGain => "measured objective gain",
            ObjectiveGainVerdict::ResourceOrMetricTradeoff => "resource or metric tradeoff",
            ObjectiveGainVerdict::ObjectiveRegression => "objective regression",
        };
        let metric_json = |measurement: crate::precision::PrecisionMeasurement| {
            serde_json::json!({
                "rmse": measurement.rmse,
                "max_absolute_error": measurement.max_absolute_error,
                "clamped_channel_events": measurement.clamped_channel_events,
                "reference_gradient_events": measurement.reference_gradient_events,
                "retained_gradient_events": measurement.retained_gradient_events,
            })
        };
        let assessment_json = |assessment: &ArtisticGainAssessment| {
            serde_json::json!({
                "verdict": verdict_str(assessment.verdict),
                "rmse_reduction": assessment.rmse_reduction,
                "max_error_reduction": assessment.max_error_reduction,
                "clamped_events_avoided": assessment.clamped_events_avoided,
                "gradients_recovered": assessment.gradients_recovered,
            })
        };
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let receipt = serde_json::json!({
            "schema": "collide-o-scope-full16-history-candidate-receipt/1",
            "command": "cargo test --locked gpu_full16_history_candidate_measures_temporal_gain_and_writes_the_receipt -- --ignored --nocapture",
            "measured_at": {
                "commit": git(&["rev-parse", "HEAD"]),
                "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
                // A dirty tree names its parent commit; say so rather than
                // implying the named commit alone produced the numbers.
                "tree": match git(&["status", "--porcelain"]).as_str() {
                    "unknown" => "unknown",
                    "" => "clean",
                    _ => "dirty",
                },
            },
            "adapter": {
                "name": adapter_info.name,
                "backend": format!("{:?}", adapter_info.backend),
                "device_type": format!("{:?}", adapter_info.device_type),
                "driver": adapter_info.driver,
                "driver_info": adapter_info.driver_info,
            },
            "scope": "ExperimentalFull16History remains evaluation-only: the candidate host is constructed by this fixture alone, presentation and dither are unchanged, and the settled AdvancedWorking16HistoryCompat8 default does not move (pinned by the M6 receipt's output SHAs).",
            "dimensions": dimensions,
            "resource_delta": {
                "fixture_additional_bytes": delta.additional_bytes,
                "documented_1080p_additional_bytes": 207_360_000_u64,
                "documented_1080p_additional_mib": "197.753906",
                "surfaces": "all 25 temporal surfaces (24-layer clean-history ring plus recursive feedback) widen from 4 to 8 bytes per pixel; no other surface changes",
            },
            "feedback_recursion": {
                "frames": TEMPORAL_FRAMES,
                "feedback": temporal_params.feedback,
                "workload": "production feedback shader over a moving pulse; analytic f32 max/decay reference",
                "settled_compat8_history": metric_json(settled_feedback),
                "full16_history_candidate": metric_json(candidate_feedback),
                "assessment": assessment_json(&feedback_assessment),
            },
            "clean_history_storage_fidelity": {
                "workload": "one frame recorded into ring layer 0 through the production no-dither conversion pipeline and read back through the production copy pipeline; reference is the exact f32 frame",
                "settled_compat8_history": metric_json(settled_ring),
                "full16_history_candidate": metric_json(candidate_ring),
                "assessment": assessment_json(&ring_assessment),
            },
        });
        std::fs::write(
            "docs/evidence/full16-history-candidate-receipt.json",
            format!("{}\n", serde_json::to_string_pretty(&receipt).unwrap()),
        )
        .expect("write the Full-16 candidate receipt");
        println!(
            "FULL16_CANDIDATE_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn warmed_encode_allocates_nothing_and_zero_temporal_is_byte_exact() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for composition host test");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Composition host test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for composition host test");
        let dimensions = [2, 1];
        let host = CompositionHost::new(&device, dimensions, capacities(dimensions, true)).unwrap();

        let source_bytes = [
            0x00, 0x3c, 0x00, 0x38, 0x00, 0x34, 0x00, 0x3a, // 1,.5,.25,.75
            0x00, 0x00, 0x00, 0x34, 0x00, 0x38, 0x00, 0x3c, // 0,.25,.5,1
        ];
        let source_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Composition host exact source"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_WORKING_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &source_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * 8),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let present_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Composition host present target"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HOST_PRESENT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let present_view = present_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ping = host.surface(HostSurface::Ping);
        let pong = host.surface(HostSurface::Pong);
        let program = host.surface(HostSurface::Program);
        let group = host.surface(HostSurface::GroupScratch);
        let effect = host.prepare_effect_source(&device, &source_view);
        let copy = host.prepare_copy_source(&device, ping.view);
        let composite = host.prepare_composite_inputs(&device, ping.view, pong.view);
        let matte = host.prepare_matte_inputs(&device, group.view, ping.view, pong.view);
        let bus = host.prepare_bus_inputs(&device, ping.view, pong.view, program.view);
        let temporal = host.prepare_temporal_input(&device, &source_view, pong.view);
        let present = host.prepare_copy_source(&device, pong.view);

        host.write_effect_uniform(
            &queue,
            HostUniformSlot(0),
            &EffectPassUniforms::for_target(
                crate::effects::EffectUniforms::default(),
                crate::spatial::SpatialTransform::default(),
                (dimensions[0], dimensions[1]),
                (dimensions[0], dimensions[1]),
            ),
        )
        .unwrap();
        host.write_composite_uniform(
            &queue,
            HostUniformSlot(0),
            &HostCompositeUniforms::new(1.0, BlendMode::Normal),
        )
        .unwrap();
        host.write_matte_uniform(
            &queue,
            HostUniformSlot(0),
            &MatteCompositeUniforms {
                opacity: 1.0,
                blend_mode: 0,
                channel: 0,
                invert: 0,
                amount: 1.0,
                threshold: 0.5,
                softness: 0.1,
                donor_valid: 1,
            },
        )
        .unwrap();
        let warmed = host.allocation_snapshot();

        let padded_row = 256_u32;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Composition host exact readback"),
            size: u64::from(padded_row),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Composition host allocation test encoder"),
        });
        host.encode_effect(&mut encoder, &effect, ping.view, HostUniformSlot(0))
            .unwrap();
        host.encode_copy(&mut encoder, &copy, pong.view);
        host.encode_composite(&mut encoder, &composite, group.view, HostUniformSlot(0))
            .unwrap();
        host.encode_matte(&mut encoder, &matte, program.view, HostUniformSlot(0))
            .unwrap();
        host.encode_bus(&queue, &mut encoder, &bus, group.view, 0.4);
        host.encode_temporal(
            &queue,
            &mut encoder,
            &temporal,
            &source_texture,
            pong.texture,
            pong.view,
            &TemporalParams::default(),
            HostFrameTiming::new(1.0 / 30.0, true),
        );
        host.encode_present(&mut encoder, &present, &present_view, HOST_PRESENT_FORMAT)
            .unwrap();
        assert_eq!(host.program_history_read_index(), 0);
        assert_eq!(host.program_history_write_index(), 1);
        host.encode_stage_program_history(&mut encoder, pong.texture);
        assert_eq!(host.allocation_snapshot(), warmed);
        assert_eq!(host.temporal_history_valid(), 1);
        host.discard_frame_history();
        assert!(!host.program_history_initialized());
        assert_eq!(host.program_history_read_index(), 0);
        assert_eq!(host.temporal_history_valid(), 0);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: pong.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        receive.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        assert_eq!(&mapped[..source_bytes.len()], &source_bytes);
        drop(mapped);
        staging.unmap();

        let mut committed_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Composition host committed Program history"),
            });
        host.encode_stage_program_history(&mut committed_encoder, pong.texture);
        queue.submit(std::iter::once(committed_encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        host.commit_frame_history();
        assert!(host.program_history_initialized());
        assert_eq!(host.program_history_read_index(), 1);
        assert_eq!(host.program_history_write_index(), 0);
        host.reset_program_history();
        assert_eq!(host.program_history_read_index(), 0);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn routed_garden_gpu_uses_selected_signal_and_preserves_warm_allocation_law() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for routed Garden proof");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Routed Garden proof device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for routed Garden proof");
        let dimensions = [2, 1];
        let host =
            CompositionHost::new(&device, dimensions, capacities(dimensions, false)).unwrap();
        let ping = host.surface(HostSurface::Ping);
        let pong = host.surface(HostSurface::Pong);
        let closed_signal = host.surface(HostSurface::A);
        let open_signal = host.surface(HostSurface::B);
        let temporal = host.prepare_temporal_input(&device, ping.view, pong.view);
        let closed = host.prepare_routed_garden_input(&device, pong.view, closed_signal.view);
        let open = host.prepare_routed_garden_input(&device, pong.view, open_signal.view);
        let warmed = host.allocation_snapshot();

        let mut initialize =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        host.encode_clear(
            &mut initialize,
            closed_signal.view,
            wgpu::Color::TRANSPARENT,
        );
        host.encode_clear(&mut initialize, open_signal.view, wgpu::Color::WHITE);
        queue.submit(Some(initialize.finish()));

        let mut params = TemporalParams::default();
        params.originals.garden.amount = 1.0;
        params.originals.garden.gate = crate::temporal::RefreshGardenGate::Matte;
        params.originals.garden.threshold = 0.5;
        params.originals.garden.softness = 0.0;
        params.originals.garden.decay = 1.0;
        let timing = HostFrameTiming::new(1.0 / 30.0, true);
        let red = repeated_half_rgba([0x3c00, 0, 0, 0x3c00], 2);
        let blue = repeated_half_rgba([0, 0, 0x3c00, 0x3c00], 2);

        upload_texture_bytes(&queue, ping.texture, dimensions, 8, &red);
        let mut prime =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(host.encode_temporal_routed(
            &queue,
            &mut prime,
            &temporal,
            ping.texture,
            ping.view,
            pong.texture,
            pong.view,
            &closed,
            &params,
            timing,
        ));
        queue.submit(Some(prime.finish()));
        host.commit_temporal_frame();

        upload_texture_bytes(&queue, ping.texture, dimensions, 8, &blue);
        let mut held =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(host.encode_temporal_routed(
            &queue,
            &mut held,
            &temporal,
            ping.texture,
            ping.view,
            pong.texture,
            pong.view,
            &closed,
            &params,
            timing,
        ));
        queue.submit(Some(held.finish()));
        host.commit_temporal_frame();
        let held = advanced_linear_samples(&read_texture_bytes(
            &device,
            &queue,
            ping.texture,
            dimensions,
            8,
            "Routed Garden closed readback",
        ));
        assert_near(held[0], [1.0, 0.0, 0.0, 1.0]);

        upload_texture_bytes(&queue, ping.texture, dimensions, 8, &blue);
        let mut admitted =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(host.encode_temporal_routed(
            &queue,
            &mut admitted,
            &temporal,
            ping.texture,
            ping.view,
            pong.texture,
            pong.view,
            &open,
            &params,
            timing,
        ));
        queue.submit(Some(admitted.finish()));
        host.commit_temporal_frame();
        let admitted = advanced_linear_samples(&read_texture_bytes(
            &device,
            &queue,
            ping.texture,
            dimensions,
            8,
            "Routed Garden open readback",
        ));
        assert_near(admitted[0], [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(host.allocation_snapshot(), warmed);
    }
}
