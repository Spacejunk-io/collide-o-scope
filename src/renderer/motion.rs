//! GPU-specific motion transforms, pass accounting, and transactional memory.
//!
//! Field geometry, resource admission, packed samples, and hard caps are owned
//! by `crate::motion`. This adapter intentionally does not restate those laws.

use std::{collections::BTreeMap, fmt};

use bytemuck::Zeroable;

use crate::visual_rack::PREMULTIPLIED_BILINEAR_TEXTURE_OPS;
use crate::{
    evaluated_frame::evaluated_composition::{
        AdvancedMotionPlan, MotionFieldAttachment, MotionPlanDiagnostic,
    },
    motion::{
        CurvedShutterParams, MotionCarrier, MotionField, MotionFieldOrigin, MotionGrid,
        MotionResourcePlan, MOTION_ALGORITHM_VERSION,
    },
    spatial::{SpatialGpuUniforms, SpatialTransform},
    temporal::TemporalFrameInput,
    visual_rack::VisualScopeId,
};

pub(crate) const MOTION_VECTOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
pub(crate) const MOTION_GATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
pub(crate) const MOTION_LUMA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub(crate) const MOTION_CARRIER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const MOTION_GARDEN_SIGNAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MotionGpuFieldSpec {
    pub grid: MotionGrid,
    pub requires_luma: bool,
    pub required_as_garden_signal: bool,
}

pub(crate) struct MotionGpuFieldSource<'a> {
    pub slot: u8,
    pub view: &'a wgpu::TextureView,
}

/// One admitted motion field's primitive vector/gate ping-pong pair, both
/// parities, for a routed consumer that prebuilds bind groups.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MotionPrimitiveFieldViews<'a> {
    pub vectors: [&'a wgpu::TextureView; 2],
    pub gates: [&'a wgpu::TextureView; 2],
    /// `[width, height]` of the field's own `MotionGrid`, not the output size.
    pub grid: [u32; 2],
}

/// Which committed parity a routed consumer must read this frame, and whether
/// that parity holds a materialized field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MotionFieldReadParity {
    pub index: usize,
    pub valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MotionGpuScopeSpec {
    pub scope: VisualScopeId,
    pub render_field_slot: u8,
    pub uses_carrier: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MotionFrameInput<'a> {
    pub attachments: &'a [MotionFieldAttachment<'a>],
    /// Media acquisition is additionally held for these scopes. Program
    /// Freeze and Media Freeze remain authoritative global gates.
    pub held_scopes: &'a [VisualScopeId],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MotionRuntimeDiagnostic {
    Planned(MotionPlanDiagnostic),
    MissingCodecAttachment { scope: VisualScopeId },
    RejectedCodecAttachment { scope: VisualScopeId },
    InvalidTransform { scope: VisualScopeId },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "Main telemetry consumes this frozen snapshot")
)]
pub(crate) struct MotionRuntimeMetrics {
    /// Accepted motion-state commits since the last explicit reset/clear.
    /// Discard and Program Freeze never advance this generation.
    pub memory_generation: u64,
    pub active_field_slots: u32,
    pub persistent_carriers: u32,
    pub valid_fields: u32,
    pub valid_luma_fields: u32,
    /// Exact field currently committed in the rendered slot for this scope.
    /// A Faraday recipient reports its admitted donor field, not its own
    /// authored source decision. Invalid/unprimed fields report `None`.
    pub field_origin: MotionFieldOrigin,
    pub field_source_scope: Option<VisualScopeId>,
    pub field_source_generation: Option<u64>,
    pub field_frame_ordinal: Option<u64>,
    pub field_product_content_sha256: Option<[u8; 32]>,
    pub carrier_valid: bool,
    pub frame_staged: bool,
    pub committed_carrier_index: u8,
    pub shutter_samples: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MotionGpuError {
    EmptyPlanMismatch,
    FieldCountMismatch {
        planned: u32,
        supplied: usize,
    },
    ResourcePlanMismatch,
    CarrierCountMismatch(u32),
    FieldGridMismatch {
        expected: MotionGrid,
        actual: MotionGrid,
    },
    FieldIndex(usize),
    DuplicateFieldSource(u8),
    MissingFieldSource(u8),
    DuplicateScope(VisualScopeId),
    MissingScope(VisualScopeId),
    BindingsNotPrepared,
    FrameAlreadyStaged,
    FrameNotStaged,
    ArithmeticOverflow,
}

impl fmt::Display for MotionGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlanMismatch => {
                formatter.write_str("an exact-zero motion plan must not allocate GPU resources")
            }
            Self::FieldCountMismatch { planned, supplied } => write!(
                formatter,
                "motion plan admits {planned} fields but renderer received {supplied} field specs"
            ),
            Self::ResourcePlanMismatch => formatter.write_str(
                "motion GPU field specs do not match the canonical resource-plan byte ledger",
            ),
            Self::CarrierCountMismatch(count) => write!(
                formatter,
                "motion renderer supports exactly zero or one persistent carrier, not {count}"
            ),
            Self::FieldGridMismatch { expected, actual } => write!(
                formatter,
                "motion field grid {}x{} does not match prepared grid {}x{}",
                actual.width, actual.height, expected.width, expected.height
            ),
            Self::FieldIndex(index) => {
                write!(formatter, "motion field slot {index} is not prepared")
            }
            Self::DuplicateFieldSource(slot) => {
                write!(
                    formatter,
                    "motion field slot {slot} has duplicate source bindings"
                )
            }
            Self::MissingFieldSource(slot) => {
                write!(formatter, "motion field slot {slot} has no source binding")
            }
            Self::DuplicateScope(scope) => {
                write!(
                    formatter,
                    "motion scope {scope:?} has duplicate GPU bindings"
                )
            }
            Self::MissingScope(scope) => {
                write!(formatter, "motion scope {scope:?} has no GPU binding")
            }
            Self::BindingsNotPrepared => {
                formatter.write_str("motion composition bindings were not prepared")
            }
            Self::FrameAlreadyStaged => formatter.write_str("motion frame is already staged"),
            Self::FrameNotStaged => formatter.write_str("motion frame was not staged"),
            Self::ArithmeticOverflow => {
                formatter.write_str("motion GPU byte arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for MotionGpuError {}

struct MotionTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl MotionTexture {
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        dimensions: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
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
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

struct MotionGpuField {
    spec: MotionGpuFieldSpec,
    vectors: [MotionTexture; 2],
    gates: [MotionTexture; 2],
    luma: Option<[MotionTexture; 2]>,
    vector_upload: Vec<u8>,
    gate_upload: Vec<u8>,
    memory: MotionMemoryState,
    frame_stage: Option<MotionMemoryStage>,
    committed_field: Option<MotionAcceptedField>,
    /// Outer `Option` means a Program frame was staged; the inner value is
    /// the exact field identity that will become resident on commit.
    staged_field: Option<Option<MotionAcceptedField>>,
    bindings: Option<MotionFieldBindings>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MotionAcceptedField {
    origin: MotionFieldOrigin,
    source_scope: VisualScopeId,
    source_generation: Option<u64>,
    frame_ordinal: Option<u64>,
    product_content_sha256: Option<[u8; 32]>,
}

impl MotionAcceptedField {
    fn codec(attachment: MotionFieldAttachment<'_>) -> Self {
        Self {
            origin: MotionFieldOrigin::CodecVectors,
            source_scope: attachment.scope,
            source_generation: Some(attachment.source_generation),
            frame_ordinal: Some(attachment.frame_ordinal),
            product_content_sha256: Some(attachment.product_content_sha256),
        }
    }

    const fn lattice(origin: MotionFieldOrigin, source_scope: VisualScopeId) -> Self {
        Self {
            origin,
            source_scope,
            source_generation: None,
            frame_ordinal: None,
            product_content_sha256: None,
        }
    }
}

struct MotionFieldBindings {
    luma: wgpu::BindGroup,
    /// Indexed by the committed/read luma parity. The target is always the
    /// inactive parity, so no command can overwrite committed luma.
    lattice: [wgpu::BindGroup; 2],
}

struct MotionScopeBindings {
    render_field_slot: u8,
    /// For each field parity: current pixels, carrier parity 0, carrier parity
    /// 1. Carrier entries exist only for the one admitted transplant.
    apply: [Vec<wgpu::BindGroup>; 2],
    refresh: [wgpu::BindGroup; 2],
    apply_uniform: wgpu::Buffer,
    refresh_uniform: wgpu::Buffer,
    #[allow(dead_code, reason = "read by the telemetry snapshot seam")]
    shutter_samples: u8,
    uses_carrier: bool,
    spatial_memory: MotionSpatialMemory,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LatticeGpuUniforms {
    field_size: [u32; 2],
    source_size: [u32; 2],
    search_radius: u32,
    update_hz: u32,
    algorithm_version: u32,
    _reserved: u32,
}

const _: () = assert!(std::mem::size_of::<LatticeGpuUniforms>() == 32);

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionApplyGpuUniforms {
    transform: MotionTransformGpu,
    shutter_values: [f32; 4],
    faraday_values: [f32; 4],
    frame_values: [f32; 4],
    modes: [u32; 4],
    spatial_samples: [MotionSpatialSampleGpu; 16],
}

const _: () = assert!(std::mem::size_of::<MotionApplyGpuUniforms>() == 1_664);

/// Three affine output-to-current-image maps for one shutter instant. The
/// chromatic variants let RGB lag evaluate the authored transform at the same
/// distinct subframe times as content-field advection without adding a fourth
/// sampled texture to the pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionSpatialSampleGpu {
    red_row_0: [f32; 4],
    red_row_1: [f32; 4],
    green_row_0: [f32; 4],
    green_row_1: [f32; 4],
    blue_row_0: [f32; 4],
    blue_row_1: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<MotionSpatialSampleGpu>() == 96);

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RefreshGpuUniforms {
    faraday_values: [f32; 4],
    gate_values: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<RefreshGpuUniforms>() == 32);

impl MotionGpuField {
    fn new(device: &wgpu::Device, spec: MotionGpuFieldSpec) -> Result<Self, MotionGpuError> {
        let vector_len = usize::try_from(
            spec.grid
                .vector_count
                .checked_mul(4)
                .ok_or(MotionGpuError::ArithmeticOverflow)?,
        )
        .map_err(|_| MotionGpuError::ArithmeticOverflow)?;
        let gate_len = usize::try_from(
            spec.grid
                .vector_count
                .checked_mul(2)
                .ok_or(MotionGpuError::ArithmeticOverflow)?,
        )
        .map_err(|_| MotionGpuError::ArithmeticOverflow)?;
        let dimensions = [spec.grid.width, spec.grid.height];
        let field_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let vectors = std::array::from_fn(|_| {
            MotionTexture::new(
                device,
                "Motion vector RG16Float parity",
                dimensions,
                MOTION_VECTOR_FORMAT,
                field_usage,
            )
        });
        let gates = std::array::from_fn(|_| {
            MotionTexture::new(
                device,
                "Motion confidence/visibility RG8 parity",
                dimensions,
                MOTION_GATE_FORMAT,
                field_usage,
            )
        });
        let luma = spec.requires_luma.then(|| {
            std::array::from_fn(|_| {
                MotionTexture::new(
                    device,
                    "Motion lattice luma R8 parity",
                    dimensions,
                    MOTION_LUMA_FORMAT,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
            })
        });
        Ok(Self {
            spec,
            vectors,
            gates,
            luma,
            vector_upload: vec![0; vector_len],
            gate_upload: vec![0; gate_len],
            memory: MotionMemoryState::default(),
            frame_stage: None,
            committed_field: None,
            staged_field: None,
            bindings: None,
        })
    }

    fn upload_field(
        &mut self,
        queue: &wgpu::Queue,
        field: &MotionField,
        write_index: u8,
    ) -> Result<(), MotionGpuError> {
        let actual = field.grid();
        if actual != self.spec.grid {
            return Err(MotionGpuError::FieldGridMismatch {
                expected: self.spec.grid,
                actual,
            });
        }
        encode_field_upload(field, &mut self.vector_upload, &mut self.gate_upload)?;
        let extent = wgpu::Extent3d {
            width: self.spec.grid.width,
            height: self.spec.grid.height,
            depth_or_array_layers: 1,
        };
        let parity = usize::from(write_index & 1);
        queue.write_texture(
            self.vectors[parity].texture.as_image_copy(),
            &self.vector_upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.spec.grid.width * 4),
                rows_per_image: Some(self.spec.grid.height),
            },
            extent,
        );
        queue.write_texture(
            self.gates[parity].texture.as_image_copy(),
            &self.gate_upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.spec.grid.width * 2),
                rows_per_image: Some(self.spec.grid.height),
            },
            extent,
        );
        Ok(())
    }
}

/// Lazily created only for a non-zero canonical plan. The exact path receives
/// `None`, therefore creates neither low-resolution surfaces nor a carrier.
pub(crate) struct MotionGpuResources {
    fields: Vec<MotionGpuField>,
    carriers: Option<[MotionTexture; 2]>,
    garden_signal: Option<MotionTexture>,
    garden_signal_bindings: Option<[wgpu::BindGroup; 2]>,
    garden_signal_field_slot: Option<u8>,
    output_dimensions: [u32; 2],
    #[allow(dead_code, reason = "read by resource and telemetry snapshot seams")]
    plan: MotionResourcePlan,
    pipelines: MotionPipelines,
    scopes: BTreeMap<VisualScopeId, MotionScopeBindings>,
    field_scopes: BTreeMap<VisualScopeId, u8>,
    carrier_memory: MotionMemoryState,
    carrier_stage: Option<MotionMemoryStage>,
    program_advances: bool,
    runtime_diagnostics: Vec<MotionRuntimeDiagnostic>,
    memory_generation: u64,
}

struct MotionPipelines {
    luma: wgpu::RenderPipeline,
    lattice: wgpu::RenderPipeline,
    apply: wgpu::RenderPipeline,
    refresh: wgpu::RenderPipeline,
    garden_signal: wgpu::RenderPipeline,
    luma_layout: wgpu::BindGroupLayout,
    lattice_layout: wgpu::BindGroupLayout,
    image_layout: wgpu::BindGroupLayout,
    garden_signal_layout: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
}

impl MotionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Motion linear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Motion nearest clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let luma_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion luma BGL"),
            entries: &[sampled_texture_entry(0), filtering_sampler_entry(1)],
        });
        let lattice_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion lattice BGL"),
            entries: &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                filtering_sampler_entry(2),
                uniform_entry(3),
            ],
        });
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion apply/refresh BGL"),
            entries: &[
                sampled_texture_entry(0),
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                filtering_sampler_entry(3),
                nonfiltering_sampler_entry(4),
                uniform_entry(5),
            ],
        });
        let garden_signal_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Motion Garden signal BGL"),
                entries: &[
                    sampled_texture_entry(0),
                    sampled_texture_entry(1),
                    nonfiltering_sampler_entry(2),
                ],
            });
        let luma_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion luma shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_luma.wgsl").into()),
        });
        let lattice_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion lattice shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_lattice.wgsl").into()),
        });
        let apply_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion apply shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_apply.wgsl").into()),
        });
        let refresh_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion refresh shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_refresh.wgsl").into()),
        });
        let garden_signal_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion Garden signal shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/motion_garden_signal.wgsl").into(),
            ),
        });
        let luma = create_motion_pipeline(
            device,
            "Motion luma pipeline",
            &luma_layout,
            &luma_module,
            &[MOTION_LUMA_FORMAT],
        );
        let lattice = create_motion_pipeline(
            device,
            "Motion lattice pipeline",
            &lattice_layout,
            &lattice_module,
            &[MOTION_VECTOR_FORMAT, MOTION_GATE_FORMAT],
        );
        let apply = create_motion_pipeline(
            device,
            "Motion curved shutter/Faraday advection pipeline",
            &image_layout,
            &apply_module,
            &[MOTION_CARRIER_FORMAT],
        );
        let refresh = create_motion_pipeline(
            device,
            "Motion Faraday refresh pipeline",
            &image_layout,
            &refresh_module,
            &[MOTION_CARRIER_FORMAT],
        );
        let garden_signal = create_motion_pipeline(
            device,
            "Motion Garden signal pipeline",
            &garden_signal_layout,
            &garden_signal_module,
            &[MOTION_GARDEN_SIGNAL_FORMAT],
        );
        Self {
            luma,
            lattice,
            apply,
            refresh,
            garden_signal,
            luma_layout,
            lattice_layout,
            image_layout,
            garden_signal_layout,
            linear_sampler,
            nearest_sampler,
        }
    }
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

fn filtering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn nonfiltering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_motion_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    target_formats: &[wgpu::TextureFormat],
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let targets = target_formats
        .iter()
        .copied()
        .map(|format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect::<Vec<_>>();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl MotionGpuResources {
    pub(crate) fn prepare(
        device: &wgpu::Device,
        plan: MotionResourcePlan,
        field_specs: &[MotionGpuFieldSpec],
        output_dimensions: [u32; 2],
    ) -> Result<Option<Self>, MotionGpuError> {
        if plan == MotionResourcePlan::default() {
            return if field_specs.is_empty() {
                Ok(None)
            } else {
                Err(MotionGpuError::EmptyPlanMismatch)
            };
        }
        if usize::try_from(plan.active_field_slots).ok() != Some(field_specs.len()) {
            return Err(MotionGpuError::FieldCountMismatch {
                planned: plan.active_field_slots,
                supplied: field_specs.len(),
            });
        }
        if plan.persistent_carriers > 1 {
            return Err(MotionGpuError::CarrierCountMismatch(
                plan.persistent_carriers,
            ));
        }
        let garden_signal_slots = field_specs
            .iter()
            .enumerate()
            .filter_map(|(slot, spec)| spec.required_as_garden_signal.then_some(slot))
            .collect::<Vec<_>>();
        if u32::try_from(garden_signal_slots.len()).ok() != Some(plan.active_garden_signals)
            || plan.active_garden_signals > 1
        {
            return Err(MotionGpuError::ResourcePlanMismatch);
        }
        let expected_garden_signal_bytes = garden_signal_slots
            .first()
            .map_or(0, |slot| field_specs[*slot].grid.vector_count);
        if expected_garden_signal_bytes != plan.garden_signal_bytes {
            return Err(MotionGpuError::ResourcePlanMismatch);
        }
        let (vector_bytes, gate_bytes, luma_bytes) = field_specs
            .iter()
            .try_fold((0_u64, 0_u64, 0_u64), |(vectors, gates, luma), spec| {
                let count = spec.grid.vector_count;
                Some((
                    vectors.checked_add(count.checked_mul(8)?)?,
                    gates.checked_add(count.checked_mul(4)?)?,
                    luma.checked_add(if spec.requires_luma {
                        count.checked_mul(2)?
                    } else {
                        0
                    })?,
                ))
            })
            .ok_or(MotionGpuError::ArithmeticOverflow)?;
        if (vector_bytes, gate_bytes, luma_bytes)
            != (plan.vector_bytes, plan.gate_bytes, plan.luma_bytes)
        {
            return Err(MotionGpuError::ResourcePlanMismatch);
        }
        let fields = field_specs
            .iter()
            .copied()
            .map(|spec| MotionGpuField::new(device, spec))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_carrier_bytes = if plan.persistent_carriers == 1 {
            u64::from(output_dimensions[0])
                .checked_mul(u64::from(output_dimensions[1]))
                .and_then(|pixels| pixels.checked_mul(16))
                .ok_or(MotionGpuError::ArithmeticOverflow)?
        } else {
            0
        };
        if plan.carrier_bytes != expected_carrier_bytes {
            return Err(MotionGpuError::ResourcePlanMismatch);
        }
        let carriers = (plan.persistent_carriers == 1).then(|| {
            std::array::from_fn(|_| {
                MotionTexture::new(
                    device,
                    "Faraday transactional carrier RGBA16Float parity",
                    output_dimensions,
                    MOTION_CARRIER_FORMAT,
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
            })
        });
        let pipelines = MotionPipelines::new(device);
        let garden_signal_field_slot = garden_signal_slots
            .first()
            .copied()
            .map(|slot| u8::try_from(slot).map_err(|_| MotionGpuError::ArithmeticOverflow))
            .transpose()?;
        let garden_signal = garden_signal_field_slot.map(|slot| {
            let grid = fields[usize::from(slot)].spec.grid;
            MotionTexture::new(
                device,
                "Motion routed Garden signal R8",
                [grid.width, grid.height],
                MOTION_GARDEN_SIGNAL_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            )
        });
        let garden_signal_bindings = garden_signal_field_slot.map(|slot| {
            let field = &fields[usize::from(slot)];
            std::array::from_fn(|parity| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Motion prepared routed Garden signal BG"),
                    layout: &pipelines.garden_signal_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &field.vectors[parity].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&field.gates[parity].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&pipelines.nearest_sampler),
                        },
                    ],
                })
            })
        });
        Ok(Some(Self {
            fields,
            carriers,
            garden_signal,
            garden_signal_bindings,
            garden_signal_field_slot,
            output_dimensions,
            plan,
            pipelines,
            scopes: BTreeMap::new(),
            field_scopes: BTreeMap::new(),
            carrier_memory: MotionMemoryState::default(),
            carrier_stage: None,
            program_advances: false,
            runtime_diagnostics: Vec::new(),
            memory_generation: 0,
        }))
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "canonical-budget GPU fixtures inspect this ledger"
        )
    )]
    pub(crate) fn plan(&self) -> MotionResourcePlan {
        self.plan
    }

    pub(crate) fn garden_signal_view(&self) -> Option<&wgpu::TextureView> {
        self.garden_signal.as_ref().map(|signal| &signal.view)
    }

    /// Both committed ping/pong parities of one admitted field's primitive
    /// vector/gate pair, plus the field's own `MotionGrid` extent.
    ///
    /// A routed consumer outside motion rendering — the dedicated Symmetry
    /// Field is the first — prebuilds a bind group for *every* parity here and
    /// selects the committed one at encode through
    /// [`MotionGpuResources::field_read_parity`]. Handing out only the
    /// currently committed view would force a rebuild on every ping/pong swap,
    /// which the warm-encode contract forbids.
    ///
    /// `slot` is an admitted field slot, which routed consumers must obtain
    /// through `EvaluatedMotionScopePlan::admitted_field_slot` so an admitted
    /// Faraday transplant cannot desync them from motion rendering.
    pub(crate) fn field_primitive_views(&self, slot: u8) -> Option<MotionPrimitiveFieldViews<'_>> {
        let field = self.fields.get(usize::from(slot))?;
        Some(MotionPrimitiveFieldViews {
            vectors: std::array::from_fn(|parity| &field.vectors[parity].view),
            gates: std::array::from_fn(|parity| &field.gates[parity].view),
            grid: [field.spec.grid.width, field.spec.grid.height],
        })
    }

    /// The committed parity a routed consumer must read for one field slot this
    /// frame, and whether that parity actually holds a materialized field.
    ///
    /// This is `MotionMemoryStage::render_field_index` — the exact index
    /// `encode_garden_signal` and `encode_scope` render from — so a routed
    /// consumer observes the same field motion rendering wrote. `None` means
    /// the slot does not exist or no frame is staged; `valid == false` means the
    /// parity exists but nothing has been written into it yet, which is an
    /// honest zero rather than a stale or unrelated field.
    pub(crate) fn field_read_parity(&self, slot: u8) -> Option<MotionFieldReadParity> {
        let field = self.fields.get(usize::from(slot))?;
        let stage = field.frame_stage?;
        Some(MotionFieldReadParity {
            index: usize::from(stage.render_field_index),
            valid: stage.render_field_valid,
        })
    }

    /// Materialize the routed scalar from the same staged field parity used by
    /// motion rendering. An unavailable first lattice observation is an honest
    /// closed (zero) signal, never an unrelated fallback.
    pub(crate) fn encode_garden_signal(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), MotionGpuError> {
        let (Some(slot), Some(signal), Some(bindings)) = (
            self.garden_signal_field_slot,
            &self.garden_signal,
            &self.garden_signal_bindings,
        ) else {
            return Ok(());
        };
        if !self.program_advances {
            return Ok(());
        }
        let field = self
            .fields
            .get(usize::from(slot))
            .ok_or(MotionGpuError::FieldIndex(usize::from(slot)))?;
        let stage = field.frame_stage.ok_or(MotionGpuError::FrameNotStaged)?;
        if !stage.render_field_valid {
            clear_target(encoder, &signal.view, wgpu::Color::TRANSPARENT);
            return Ok(());
        }
        encode_single_target_pass(
            encoder,
            "Motion routed Garden signal",
            &self.pipelines.garden_signal,
            &bindings[usize::from(stage.render_field_index)],
            &signal.view,
            None,
        );
        Ok(())
    }

    /// Builds every source/field/carrier bind group once. Warm frames only
    /// update fixed uniforms and queue-write an inactive codec parity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_composition_bindings(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &AdvancedMotionPlan,
        field_sources: &[MotionGpuFieldSource<'_>],
        scope_specs: &[MotionGpuScopeSpec],
        current: &wgpu::TextureView,
        advected: &wgpu::TextureView,
    ) -> Result<(), MotionGpuError> {
        let mut source_lookup = BTreeMap::new();
        for source in field_sources {
            if source_lookup.insert(source.slot, source.view).is_some() {
                return Err(MotionGpuError::DuplicateFieldSource(source.slot));
            }
        }
        for field_plan in plan.fields() {
            self.field_scopes.insert(field_plan.scope, field_plan.slot);
            let field = self
                .fields
                .get_mut(usize::from(field_plan.slot))
                .ok_or(MotionGpuError::FieldIndex(usize::from(field_plan.slot)))?;
            if !field.spec.requires_luma {
                continue;
            }
            let source = source_lookup
                .remove(&field_plan.slot)
                .ok_or(MotionGpuError::MissingFieldSource(field_plan.slot))?;
            let luma_uniform = LatticeGpuUniforms {
                field_size: [field.spec.grid.width, field.spec.grid.height],
                source_size: field_plan.source_dimensions,
                search_radius: plan.scope(field_plan.scope).map_or(0, |scope| {
                    scope.params.lattice_quality.search_radius() as u32
                }),
                update_hz: plan
                    .scope(field_plan.scope)
                    .map_or(1, |scope| scope.params.lattice_quality.update_hz()),
                algorithm_version: u32::from(MOTION_ALGORITHM_VERSION),
                _reserved: 0,
            };
            let lattice_uniform =
                fixed_uniform_buffer(device, queue, "Motion lattice uniform", &luma_uniform);
            let luma = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Motion prepared luma source BG"),
                layout: &self.pipelines.luma_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.pipelines.linear_sampler),
                    },
                ],
            });
            let luma_views = field
                .luma
                .as_ref()
                .ok_or(MotionGpuError::ResourcePlanMismatch)?;
            let lattice = std::array::from_fn(|read_index| {
                let write_index = 1 - read_index;
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Motion prepared lattice parity BG"),
                    layout: &self.pipelines.lattice_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &luma_views[write_index].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &luma_views[read_index].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &self.pipelines.linear_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: lattice_uniform.as_entire_binding(),
                        },
                    ],
                })
            });
            field.bindings = Some(MotionFieldBindings { luma, lattice });
        }
        if !source_lookup.is_empty() {
            return Err(MotionGpuError::ResourcePlanMismatch);
        }

        for spec in scope_specs {
            if self.scopes.contains_key(&spec.scope) {
                return Err(MotionGpuError::DuplicateScope(spec.scope));
            }
            let field = self.fields.get(usize::from(spec.render_field_slot)).ok_or(
                MotionGpuError::FieldIndex(usize::from(spec.render_field_slot)),
            )?;
            let apply_uniform = fixed_uniform_buffer(
                device,
                queue,
                "Motion apply uniform",
                &MotionApplyGpuUniforms::zeroed(),
            );
            let refresh_uniform = fixed_uniform_buffer(
                device,
                queue,
                "Motion refresh uniform",
                &RefreshGpuUniforms::zeroed(),
            );
            let apply = std::array::from_fn(|field_parity| {
                let mut sources = Vec::with_capacity(1 + self.carriers.as_ref().map_or(0, |_| 2));
                sources.push(current);
                if let Some(carriers) = &self.carriers {
                    sources.extend(carriers.iter().map(|carrier| &carrier.view));
                }
                sources
                    .into_iter()
                    .map(|carrier| {
                        create_image_bind_group(
                            device,
                            &self.pipelines,
                            carrier,
                            &field.vectors[field_parity].view,
                            &field.gates[field_parity].view,
                            &apply_uniform,
                            "Motion prepared apply BG",
                        )
                    })
                    .collect()
            });
            let refresh = std::array::from_fn(|field_parity| {
                create_image_bind_group(
                    device,
                    &self.pipelines,
                    current,
                    advected,
                    &field.gates[field_parity].view,
                    &refresh_uniform,
                    "Motion prepared refresh BG",
                )
            });
            self.scopes.insert(
                spec.scope,
                MotionScopeBindings {
                    render_field_slot: spec.render_field_slot,
                    apply,
                    refresh,
                    apply_uniform,
                    refresh_uniform,
                    shutter_samples: plan.scope(spec.scope).map_or(0, |scope| {
                        if scope.params.shutter.is_exact_zero() {
                            0
                        } else {
                            scope.params.shutter.quality.sample_count()
                        }
                    }),
                    uses_carrier: spec.uses_carrier,
                    spatial_memory: MotionSpatialMemory::default(),
                },
            );
        }
        self.runtime_diagnostics = Vec::with_capacity(
            plan.diagnostics()
                .len()
                .saturating_add(plan.fields().len().saturating_mul(2))
                .saturating_add(plan.scopes().len()),
        );
        self.runtime_diagnostics.extend(
            plan.diagnostics()
                .iter()
                .copied()
                .map(MotionRuntimeDiagnostic::Planned),
        );
        Ok(())
    }

    pub(crate) fn begin_frame(
        &mut self,
        queue: &wgpu::Queue,
        plan: &AdvancedMotionPlan,
        temporal: TemporalFrameInput,
        input: MotionFrameInput<'_>,
    ) -> Result<(), MotionGpuError> {
        if self
            .fields
            .iter()
            .any(|field| field.frame_stage.is_some() || field.staged_field.is_some())
            || self.carrier_stage.is_some()
            || self
                .scopes
                .values()
                .any(|bindings| bindings.spatial_memory.staged.is_some())
        {
            return Err(MotionGpuError::FrameAlreadyStaged);
        }
        let needs_scope_bindings = plan.scopes().iter().any(|scope| {
            !scope.params.is_exact_zero()
                && (scope.transplant_admitted || scope.field_slot.is_some())
        });
        if self.scopes.is_empty() && needs_scope_bindings {
            return Err(MotionGpuError::BindingsNotPrepared);
        }
        self.runtime_diagnostics.clear();
        self.runtime_diagnostics.extend(
            plan.diagnostics()
                .iter()
                .copied()
                .map(MotionRuntimeDiagnostic::Planned),
        );
        self.program_advances = temporal.freeze.program_advances();
        for field_plan in plan.fields() {
            let field = self
                .fields
                .get_mut(usize::from(field_plan.slot))
                .ok_or(MotionGpuError::FieldIndex(usize::from(field_plan.slot)))?;
            let acquisition_advances =
                temporal.freeze.media_advances() && !input.held_scopes.contains(&field_plan.scope);
            let lattice = matches!(
                field_plan.source.origin,
                MotionFieldOrigin::Lattice | MotionFieldOrigin::LatticeFallback
            );
            let mut accepted_attachment = None;
            if field_plan.source.origin == MotionFieldOrigin::CodecVectors {
                for attachment in input
                    .attachments
                    .iter()
                    .copied()
                    .filter(|attachment| attachment.scope == field_plan.scope)
                {
                    if field_plan.accepts(attachment) && accepted_attachment.is_none() {
                        accepted_attachment = Some(attachment);
                    } else {
                        self.runtime_diagnostics.push(
                            MotionRuntimeDiagnostic::RejectedCodecAttachment {
                                scope: field_plan.scope,
                            },
                        );
                    }
                }
                if accepted_attachment.is_none() {
                    self.runtime_diagnostics.push(
                        MotionRuntimeDiagnostic::MissingCodecAttachment {
                            scope: field_plan.scope,
                        },
                    );
                }
            }
            let stage = field.memory.stage(
                temporal.delta_seconds,
                plan.scope(field_plan.scope)
                    .map_or(1, |scope| scope.params.lattice_quality.update_hz()),
                self.program_advances,
                acquisition_advances,
                lattice,
                accepted_attachment.is_some(),
                false,
            );
            if self.program_advances && acquisition_advances {
                if let Some(attachment) = accepted_attachment {
                    field.upload_field(queue, attachment.field, stage.write_field_index)?;
                }
            }
            let mut staged_field = field.committed_field;
            if self.program_advances && acquisition_advances {
                if let Some(attachment) = accepted_attachment {
                    staged_field = Some(MotionAcceptedField::codec(attachment));
                } else if lattice && stage.update_lattice && stage.render_field_valid {
                    staged_field = Some(MotionAcceptedField::lattice(
                        field_plan.source.origin,
                        field_plan.scope,
                    ));
                }
            }
            field.frame_stage = self.program_advances.then_some(stage);
            field.staged_field = self.program_advances.then_some(staged_field);
        }
        for attachment in input.attachments.iter().copied() {
            if !plan
                .fields()
                .iter()
                .copied()
                .any(|field| field.accepts(attachment))
                && !self.runtime_diagnostics.iter().any(|diagnostic| {
                    matches!(
                        diagnostic,
                        MotionRuntimeDiagnostic::RejectedCodecAttachment { scope }
                            if *scope == attachment.scope
                    )
                })
            {
                self.runtime_diagnostics
                    .push(MotionRuntimeDiagnostic::RejectedCodecAttachment {
                        scope: attachment.scope,
                    });
            }
        }
        if self.carriers.is_some() {
            let stage = self.carrier_memory.stage(
                temporal.delta_seconds,
                1,
                self.program_advances,
                false,
                false,
                false,
                true,
            );
            self.carrier_stage = self.program_advances.then_some(stage);
        }
        self.write_scope_uniforms(queue, plan, temporal.delta_seconds)?;
        Ok(())
    }

    fn write_scope_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        plan: &AdvancedMotionPlan,
        delta_seconds: f32,
    ) -> Result<(), MotionGpuError> {
        for scope in plan.scopes() {
            let Some(render_field_slot) = self
                .scopes
                .get(&scope.scope)
                .map(|bindings| bindings.render_field_slot)
            else {
                if scope.params.is_exact_zero()
                    || (!scope.transplant_admitted && scope.field_slot.is_none())
                {
                    continue;
                }
                return Err(MotionGpuError::MissingScope(scope.scope));
            };
            let donor = if scope.transplant_admitted {
                scope
                    .donor_scope
                    .and_then(|donor| plan.scope(donor))
                    .ok_or(MotionGpuError::MissingScope(scope.scope))?
            } else {
                scope
            };
            let transform = MotionTransformGpu::between(donor.spatial, scope.spatial);
            if transform.is_none() {
                self.runtime_diagnostics
                    .push(MotionRuntimeDiagnostic::InvalidTransform { scope: scope.scope });
            }
            let faraday = scope.params.transplant;
            let shutter = scope.params.shutter;
            let field_valid = self.fields[usize::from(render_field_slot)]
                .frame_stage
                .is_some_and(|stage| stage.render_field_valid);
            let bindings = self
                .scopes
                .get_mut(&scope.scope)
                .ok_or(MotionGpuError::MissingScope(scope.scope))?;
            let spatial_stage = bindings
                .spatial_memory
                .stage(scope.transform, self.program_advances);
            let spatial_samples = motion_spatial_samples(
                spatial_stage.previous,
                spatial_stage.current,
                scope.source_dimensions,
                self.output_dimensions,
                shutter,
            );
            if spatial_samples.is_none() {
                self.runtime_diagnostics
                    .push(MotionRuntimeDiagnostic::InvalidTransform { scope: scope.scope });
            }
            let active = self.program_advances
                && transform.is_some()
                && spatial_samples.is_some()
                && (field_valid || !shutter.is_exact_zero());
            let uniforms = MotionApplyGpuUniforms {
                transform: transform.unwrap_or_else(MotionTransformGpu::identity),
                shutter_values: [
                    shutter.angle_degrees,
                    shutter.phase,
                    shutter.curvature,
                    shutter.chromatic_lag,
                ],
                faraday_values: [
                    if scope.transplant_admitted {
                        faraday.amount
                    } else {
                        0.0
                    },
                    faraday.confidence_threshold,
                    faraday.confidence_softness,
                    faraday.occlusion,
                ],
                frame_values: [delta_seconds.max(0.0), 0.0, 0.0, 0.0],
                modes: [
                    u32::from(active),
                    u32::from(if shutter.is_exact_zero() {
                        1
                    } else {
                        shutter.quality.sample_count()
                    }),
                    0,
                    u32::from(field_valid),
                ],
                spatial_samples: spatial_samples.unwrap_or_else(identity_spatial_samples),
            };
            queue.write_buffer(&bindings.apply_uniform, 0, bytemuck::bytes_of(&uniforms));
            queue.write_buffer(
                &bindings.refresh_uniform,
                0,
                bytemuck::bytes_of(&RefreshGpuUniforms {
                    faraday_values: [
                        if scope.transplant_admitted {
                            faraday.amount
                        } else {
                            0.0
                        },
                        faraday.refresh,
                        faraday.decay,
                        faraday.occlusion,
                    ],
                    gate_values: [
                        faraday.confidence_threshold,
                        faraday.confidence_softness,
                        0.0,
                        0.0,
                    ],
                }),
            );
        }
        Ok(())
    }

    pub(crate) fn encode_field_scope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedMotionPlan,
        scope: VisualScopeId,
    ) -> Result<(), MotionGpuError> {
        if !self.program_advances {
            return Ok(());
        }
        let Some(field_plan) = plan.fields().iter().find(|field| field.scope == scope) else {
            return Ok(());
        };
        let field = &self.fields[usize::from(field_plan.slot)];
        let stage = field.frame_stage.ok_or(MotionGpuError::FrameNotStaged)?;
        if !stage.update_lattice {
            return Ok(());
        }
        let bindings = field
            .bindings
            .as_ref()
            .ok_or(MotionGpuError::BindingsNotPrepared)?;
        let luma = field
            .luma
            .as_ref()
            .ok_or(MotionGpuError::ResourcePlanMismatch)?;
        encode_single_target_pass(
            encoder,
            "Motion luma acquisition",
            &self.pipelines.luma,
            &bindings.luma,
            &luma[usize::from(stage.write_luma_index)].view,
            None,
        );
        if stage.luma_read_valid {
            encode_lattice_pass(
                encoder,
                &self.pipelines.lattice,
                &bindings.lattice[usize::from(stage.read_luma_index)],
                &field.vectors[usize::from(stage.write_field_index)].view,
                &field.gates[usize::from(stage.write_field_index)].view,
            );
        }
        Ok(())
    }

    /// Applies a scope after its existing host effects. `refresh_target` is
    /// existing full-frame host scratch; the only persistent full-frame motion
    /// allocation is the admitted two-parity Faraday carrier.
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit host scratch textures make aliasing and persistence auditable"
    )]
    pub(crate) fn encode_scope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedMotionPlan,
        scope_id: VisualScopeId,
        current: &wgpu::Texture,
        advected: &wgpu::Texture,
        advected_view: &wgpu::TextureView,
        refresh_target: &wgpu::Texture,
        refresh_target_view: &wgpu::TextureView,
        dimensions: [u32; 2],
    ) -> Result<(), MotionGpuError> {
        if !self.program_advances {
            return Ok(());
        }
        let Some(scope) = plan.scope(scope_id) else {
            return Ok(());
        };
        let Some(bindings) = self.scopes.get(&scope_id) else {
            return Ok(());
        };
        let field = &self.fields[usize::from(bindings.render_field_slot)];
        let field_stage = field.frame_stage.ok_or(MotionGpuError::FrameNotStaged)?;
        let faraday_active = scope.transplant_admitted;
        let shutter_active = !scope.params.shutter.is_exact_zero();
        let carrier_stage = self.carrier_stage;
        if !field_stage.render_field_valid {
            if shutter_active {
                encode_single_target_pass(
                    encoder,
                    "Motion transform-only curved shutter",
                    &self.pipelines.apply,
                    &bindings.apply[usize::from(field_stage.render_field_index)][0],
                    advected_view,
                    None,
                );
                copy_texture(encoder, advected, current, dimensions);
            }
            if faraday_active {
                self.encode_unadvected_carrier(
                    encoder,
                    scope.params.transplant.carrier,
                    current,
                    dimensions,
                )?;
            }
            return Ok(());
        }
        let field_parity = usize::from(field_stage.render_field_index);
        let carrier_source = if faraday_active {
            let stage = carrier_stage.ok_or(MotionGpuError::FrameNotStaged)?;
            if stage.carrier_read_valid {
                1 + usize::from(stage.read_carrier_index)
            } else {
                match scope.params.transplant.carrier {
                    MotionCarrier::FirstSourceFrame => 0,
                    MotionCarrier::Transparent | MotionCarrier::Black => {
                        let carriers = self
                            .carriers
                            .as_ref()
                            .ok_or(MotionGpuError::ResourcePlanMismatch)?;
                        clear_target(
                            encoder,
                            &carriers[usize::from(stage.read_carrier_index)].view,
                            if scope.params.transplant.carrier == MotionCarrier::Black {
                                wgpu::Color::BLACK
                            } else {
                                wgpu::Color::TRANSPARENT
                            },
                        );
                        1 + usize::from(stage.read_carrier_index)
                    }
                }
            }
        } else {
            0
        };
        encode_single_target_pass(
            encoder,
            "Motion curved shutter/Faraday advection",
            &self.pipelines.apply,
            &bindings.apply[field_parity][carrier_source],
            advected_view,
            None,
        );
        if faraday_active {
            let stage = carrier_stage.ok_or(MotionGpuError::FrameNotStaged)?;
            encode_single_target_pass(
                encoder,
                "Motion Faraday refresh",
                &self.pipelines.refresh,
                &bindings.refresh[field_parity],
                refresh_target_view,
                None,
            );
            copy_texture(encoder, refresh_target, current, dimensions);
            let carriers = self
                .carriers
                .as_ref()
                .ok_or(MotionGpuError::ResourcePlanMismatch)?;
            copy_texture(
                encoder,
                refresh_target,
                &carriers[usize::from(stage.write_carrier_index)].texture,
                dimensions,
            );
        } else {
            copy_texture(encoder, advected, current, dimensions);
        }
        Ok(())
    }

    fn encode_unadvected_carrier(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        policy: MotionCarrier,
        current: &wgpu::Texture,
        dimensions: [u32; 2],
    ) -> Result<(), MotionGpuError> {
        let stage = self.carrier_stage.ok_or(MotionGpuError::FrameNotStaged)?;
        let carriers = self
            .carriers
            .as_ref()
            .ok_or(MotionGpuError::ResourcePlanMismatch)?;
        let target = &carriers[usize::from(stage.write_carrier_index)];
        if stage.carrier_read_valid {
            copy_texture(
                encoder,
                &carriers[usize::from(stage.read_carrier_index)].texture,
                &target.texture,
                dimensions,
            );
        } else {
            match policy {
                MotionCarrier::FirstSourceFrame => {
                    copy_texture(encoder, current, &target.texture, dimensions)
                }
                MotionCarrier::Transparent => {
                    clear_target(encoder, &target.view, wgpu::Color::TRANSPARENT)
                }
                MotionCarrier::Black => clear_target(encoder, &target.view, wgpu::Color::BLACK),
            }
        }
        Ok(())
    }

    pub(crate) fn commit_frame(&mut self) {
        let accepted_motion_stage = self.fields.iter().any(|field| field.frame_stage.is_some())
            || self.carrier_stage.is_some()
            || self
                .scopes
                .values()
                .any(|bindings| bindings.spatial_memory.staged.is_some());
        for field in &mut self.fields {
            if field.frame_stage.take().is_some() {
                field.memory.commit();
                if let Some(accepted) = field.staged_field.take() {
                    field.committed_field = accepted;
                }
            }
        }
        if self.carrier_stage.take().is_some() {
            self.carrier_memory.commit();
        }
        for bindings in self.scopes.values_mut() {
            bindings.spatial_memory.commit();
        }
        if accepted_motion_stage {
            self.memory_generation = self.memory_generation.saturating_add(1);
        }
        self.program_advances = false;
    }

    pub(crate) fn discard_frame(&mut self) {
        for field in &mut self.fields {
            field.frame_stage = None;
            field.staged_field = None;
            field.memory.discard();
        }
        self.carrier_stage = None;
        self.carrier_memory.discard();
        for bindings in self.scopes.values_mut() {
            bindings.spatial_memory.discard();
        }
        self.program_advances = false;
    }

    #[allow(
        dead_code,
        reason = "Main/export telemetry consumes this frozen diagnostic seam"
    )]
    pub(crate) fn diagnostics(&self) -> &[MotionRuntimeDiagnostic] {
        &self.runtime_diagnostics
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "Main telemetry consumes this frozen snapshot seam"
        )
    )]
    pub(crate) fn metrics(&self, scope: VisualScopeId) -> Option<MotionRuntimeMetrics> {
        let scope_bindings = self.scopes.get(&scope);
        if scope_bindings.is_none() && !self.field_scopes.contains_key(&scope) {
            return None;
        }
        let field_slot = scope_bindings
            .map(|bindings| bindings.render_field_slot)
            .or_else(|| self.field_scopes.get(&scope).copied());
        let field_state = field_slot
            .and_then(|slot| self.fields.get(usize::from(slot)))
            .map(|field| field.memory.metrics());
        let accepted_field = field_slot
            .and_then(|slot| self.fields.get(usize::from(slot)))
            .and_then(|field| field.committed_field)
            .filter(|_| field_state.is_some_and(|state| state.field_valid));
        let uses_carrier = scope_bindings.is_some_and(|bindings| bindings.uses_carrier);
        let carrier = self.carrier_memory.metrics();
        let mut metrics = MotionRuntimeMetrics {
            memory_generation: self.memory_generation,
            active_field_slots: u32::from(field_slot.is_some()),
            persistent_carriers: u32::from(uses_carrier),
            valid_fields: u32::from(field_state.is_some_and(|state| state.field_valid)),
            valid_luma_fields: u32::from(field_state.is_some_and(|state| state.luma_valid)),
            field_origin: accepted_field.map_or(MotionFieldOrigin::None, |field| field.origin),
            field_source_scope: accepted_field.map(|field| field.source_scope),
            field_source_generation: accepted_field.and_then(|field| field.source_generation),
            field_frame_ordinal: accepted_field.and_then(|field| field.frame_ordinal),
            field_product_content_sha256: accepted_field
                .and_then(|field| field.product_content_sha256),
            carrier_valid: uses_carrier && carrier.carrier_valid,
            frame_staged: field_state.is_some_and(|state| state.frame_staged)
                || (uses_carrier && carrier.frame_staged),
            committed_carrier_index: carrier.committed_carrier_index,
            shutter_samples: scope_bindings.map_or(0, |bindings| bindings.shutter_samples),
        };
        if let Some(slot) = field_slot {
            metrics.frame_staged |= self.fields[usize::from(slot)].frame_stage.is_some();
        }
        metrics.frame_staged |= uses_carrier && self.carrier_stage.is_some();
        metrics.frame_staged |=
            scope_bindings.is_some_and(|bindings| bindings.spatial_memory.staged.is_some());
        Some(metrics)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "direct canonical upload seam is retained for embedders"
        )
    )]
    pub(crate) fn upload_codec_field(
        &mut self,
        queue: &wgpu::Queue,
        slot: usize,
        field: &MotionField,
        write_index: u8,
    ) -> Result<(), MotionGpuError> {
        self.fields
            .get_mut(slot)
            .ok_or(MotionGpuError::FieldIndex(slot))?
            .upload_field(queue, field, write_index)
    }

    pub(crate) fn reset(&mut self) {
        for field in &mut self.fields {
            field.memory.reset();
            field.frame_stage = None;
            field.committed_field = None;
            field.staged_field = None;
        }
        self.carrier_memory.reset();
        self.carrier_stage = None;
        for bindings in self.scopes.values_mut() {
            bindings.spatial_memory.reset();
        }
        self.program_advances = false;
        self.memory_generation = 0;
    }
}

fn fixed_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    value: &T,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
    buffer
}

fn create_image_bind_group(
    device: &wgpu::Device,
    pipelines: &MotionPipelines,
    first: &wgpu::TextureView,
    second: &wgpu::TextureView,
    third: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipelines.image_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(first),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(second),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(third),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&pipelines.linear_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&pipelines.nearest_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}

fn encode_single_target_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    clear: Option<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear.unwrap_or(wgpu::Color::TRANSPARENT)),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn encode_lattice_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vectors: &wgpu::TextureView,
    gates: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Motion deterministic lattice"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: vectors,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: gates,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn clear_target(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    color: wgpu::Color,
) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Motion carrier deterministic initialization"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn copy_texture(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    target: &wgpu::Texture,
    dimensions: [u32; 2],
) {
    encoder.copy_texture_to_texture(
        source.as_image_copy(),
        target.as_image_copy(),
        wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
    );
}

fn pack_unorm8(value: f32) -> u8 {
    if value.is_finite() {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        0
    }
}

fn encode_field_upload(
    field: &MotionField,
    vectors: &mut [u8],
    gates: &mut [u8],
) -> Result<(), MotionGpuError> {
    let count = usize::try_from(field.grid().vector_count)
        .map_err(|_| MotionGpuError::ArithmeticOverflow)?;
    if vectors.len()
        != count
            .checked_mul(4)
            .ok_or(MotionGpuError::ArithmeticOverflow)?
        || gates.len()
            != count
                .checked_mul(2)
                .ok_or(MotionGpuError::ArithmeticOverflow)?
    {
        return Err(MotionGpuError::ResourcePlanMismatch);
    }
    for (index, packed) in field.packed_vectors().iter().copied().enumerate() {
        let sample = packed.sample();
        let vector_offset = index * 4;
        vectors[vector_offset..vector_offset + 2]
            .copy_from_slice(&f32_to_f16_bits(sample.velocity_uv_per_second[0]).to_le_bytes());
        vectors[vector_offset + 2..vector_offset + 4]
            .copy_from_slice(&f32_to_f16_bits(sample.velocity_uv_per_second[1]).to_le_bytes());
        let gate_offset = index * 2;
        gates[gate_offset] = pack_unorm8(sample.confidence);
        gates[gate_offset + 1] = pack_unorm8(sample.visibility);
    }
    Ok(())
}

/// IEEE-754 round-to-nearest-even conversion used for RG16Float queue upload.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let normalized = mantissa | 0x80_0000;
        let shift = u32::try_from(14 - half_exponent).unwrap_or(24);
        let mut rounded = normalized >> shift;
        let remainder = normalized & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut half = (u32::try_from(half_exponent).unwrap_or(31) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 != 0) {
        half += 1;
    }
    sign | half as u16
}

/// Small CPU-only history for authored geometry. It is published by the same
/// outer transaction as field/carrier parity: a rejected command buffer can
/// never become the baseline of a later transform shutter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MotionSpatialMemory {
    committed: Option<SpatialTransform>,
    staged: Option<SpatialTransform>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MotionSpatialStage {
    previous: SpatialTransform,
    current: SpatialTransform,
}

impl MotionSpatialMemory {
    fn stage(&mut self, current: SpatialTransform, program_advances: bool) -> MotionSpatialStage {
        let current = current.sanitized();
        let previous = self.committed.unwrap_or(current);
        if program_advances {
            assert!(self.staged.is_none(), "motion spatial frame already staged");
            self.staged = Some(current);
        }
        MotionSpatialStage { previous, current }
    }

    fn commit(&mut self) {
        if let Some(staged) = self.staged.take() {
            self.committed = Some(staged);
        }
    }

    fn discard(&mut self) {
        self.staged = None;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn identity_spatial_sample() -> MotionSpatialSampleGpu {
    let row_0 = [1.0, 0.0, 0.0, 0.0];
    let row_1 = [0.0, 1.0, 0.0, 0.0];
    MotionSpatialSampleGpu {
        red_row_0: row_0,
        red_row_1: row_1,
        green_row_0: row_0,
        green_row_1: row_1,
        blue_row_0: row_0,
        blue_row_1: row_1,
    }
}

fn identity_spatial_samples() -> [MotionSpatialSampleGpu; 16] {
    [identity_spatial_sample(); 16]
}

/// Evaluates authored translation/rotation/scale (including modulation already
/// frozen into the plan) at each fixed shutter instant. Offset zero is the
/// current accepted frame and offset -1 is the prior accepted frame.
fn motion_spatial_samples(
    previous: SpatialTransform,
    current: SpatialTransform,
    source_dimensions: [u32; 2],
    output_dimensions: [u32; 2],
    shutter: CurvedShutterParams,
) -> Option<[MotionSpatialSampleGpu; 16]> {
    let current_spatial = current.gpu_uniforms(
        source_dimensions[0],
        source_dimensions[1],
        output_dimensions[0],
        output_dimensions[1],
    );
    if current_spatial.modes[2] == 0 {
        return None;
    }
    let sample_count = if shutter.is_exact_zero() {
        1
    } else {
        shutter.quality.sample_count()
    };
    let denominator = f32::from(sample_count.saturating_sub(1).max(1));
    let exposure = shutter.angle_degrees / 360.0;
    Some(std::array::from_fn(|index| {
        let time = index as f32 / denominator - 0.5 + shutter.phase * 0.5;
        let lag = shutter.chromatic_lag / denominator;
        let map = |sample_time: f32| {
            let curved = sample_time + shutter.curvature * sample_time * sample_time.abs() * 0.5;
            let sampled = extrapolate_spatial(previous, current, exposure * curved);
            let sampled_spatial = sampled.gpu_uniforms(
                source_dimensions[0],
                source_dimensions[1],
                output_dimensions[0],
                output_dimensions[1],
            );
            spatial_output_to_current(current_spatial, sampled_spatial)
                .unwrap_or(([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]))
        };
        let red = map(time - lag);
        let green = map(time);
        let blue = map(time + lag);
        MotionSpatialSampleGpu {
            red_row_0: red.0,
            red_row_1: red.1,
            green_row_0: green.0,
            green_row_1: green.1,
            blue_row_0: blue.0,
            blue_row_1: blue.1,
        }
    }))
}

fn extrapolate_spatial(
    previous: SpatialTransform,
    current: SpatialTransform,
    offset_from_current_frames: f32,
) -> SpatialTransform {
    let previous = previous.sanitized();
    let current = current.sanitized();
    let offset = if offset_from_current_frames.is_finite() {
        offset_from_current_frames
    } else {
        0.0
    };
    let linear = |before: f32, now: f32| now + (now - before) * offset;
    let degrees = |before: f32, now: f32| {
        let delta = (now - before + 180.0).rem_euclid(360.0) - 180.0;
        (now + delta * offset + 180.0).rem_euclid(360.0) - 180.0
    };
    let scale = |before: f32, now: f32| {
        if before.signum() == now.signum() && before.abs() >= 1.0e-5 && now.abs() >= 1.0e-5 {
            now.signum() * (now.abs().ln() + (now.abs().ln() - before.abs().ln()) * offset).exp()
        } else {
            linear(before, now)
        }
    };
    SpatialTransform {
        position: [
            linear(previous.position[0], current.position[0]),
            linear(previous.position[1], current.position[1]),
        ],
        scale: [
            scale(previous.scale[0], current.scale[0]),
            scale(previous.scale[1], current.scale[1]),
        ],
        anchor: [
            linear(previous.anchor[0], current.anchor[0]),
            linear(previous.anchor[1], current.anchor[1]),
        ],
        rotation_deg: degrees(previous.rotation_deg, current.rotation_deg),
        skew_deg: linear(previous.skew_deg, current.skew_deg),
        skew_axis_deg: degrees(previous.skew_axis_deg, current.skew_axis_deg),
        fit: current.fit,
        crop: [
            linear(previous.crop[0], current.crop[0]),
            linear(previous.crop[1], current.crop[1]),
            linear(previous.crop[2], current.crop[2]),
            linear(previous.crop[3], current.crop[3]),
        ],
        edge: current.edge,
        sampling: current.sampling,
    }
    .sanitized()
}

/// Maps one output pixel at an authored subframe into the already materialized
/// current-frame image. Crop participates in the affine, so an animated crop
/// follows the same source-coordinate trajectory as position/rotation/scale.
fn spatial_output_to_current(
    current: SpatialGpuUniforms,
    sampled: SpatialGpuUniforms,
) -> Option<([f32; 4], [f32; 4])> {
    if current.modes[2] == 0 || sampled.modes[2] == 0 {
        return None;
    }
    let current_inverse = [
        [current.inverse_row_0[0], current.inverse_row_0[1]],
        [current.inverse_row_1[0], current.inverse_row_1[1]],
    ];
    let current_forward = invert_2x2(current_inverse)?;
    let sampled_inverse = [
        [sampled.inverse_row_0[0], sampled.inverse_row_0[1]],
        [sampled.inverse_row_1[0], sampled.inverse_row_1[1]],
    ];
    let current_extent = [current.crop[2], current.crop[3]];
    if current_extent
        .into_iter()
        .any(|extent| !extent.is_finite() || extent.abs() <= 1.0e-12)
    {
        return None;
    }
    let extent_ratio = [
        sampled.crop[2] / current_extent[0],
        sampled.crop[3] / current_extent[1],
    ];
    let local_linear = [
        [
            sampled_inverse[0][0] * extent_ratio[0],
            sampled_inverse[0][1] * extent_ratio[0],
        ],
        [
            sampled_inverse[1][0] * extent_ratio[1],
            sampled_inverse[1][1] * extent_ratio[1],
        ],
    ];
    let local_translation = [
        (sampled.crop[0] - current.crop[0]) / current_extent[0]
            + sampled.inverse_row_0[2] * extent_ratio[0],
        (sampled.crop[1] - current.crop[1]) / current_extent[1]
            + sampled.inverse_row_1[2] * extent_ratio[1],
    ];
    let linear = multiply_2x2(current_forward, local_linear);
    let translated = [
        local_translation[0] - current.inverse_row_0[2],
        local_translation[1] - current.inverse_row_1[2],
    ];
    let translation = [
        current_forward[0][0].mul_add(translated[0], current_forward[0][1] * translated[1]),
        current_forward[1][0].mul_add(translated[0], current_forward[1][1] * translated[1]),
    ];
    linear
        .into_iter()
        .flatten()
        .chain(translation)
        .all(|value| value.is_finite())
        .then_some((
            [linear[0][0], linear[0][1], translation[0], 0.0],
            [linear[1][0], linear[1][1], translation[1], 0.0],
        ))
}

/// Coordinate payload for sampling a donor field at an output pixel and then
/// applying its vector in recipient-local coordinates. Translation applies to
/// positions only; vector conversion uses the composed 2x2 linear map.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MotionTransformGpu {
    pub output_to_donor_row_0: [f32; 4],
    pub output_to_donor_row_1: [f32; 4],
    pub donor_to_recipient_row_0: [f32; 4],
    pub donor_to_recipient_row_1: [f32; 4],
}

impl MotionTransformGpu {
    const fn identity() -> Self {
        Self {
            output_to_donor_row_0: [1.0, 0.0, 0.0, 0.0],
            output_to_donor_row_1: [0.0, 1.0, 0.0, 0.0],
            donor_to_recipient_row_0: [1.0, 0.0, 0.0, 0.0],
            donor_to_recipient_row_1: [0.0, 1.0, 0.0, 0.0],
        }
    }

    pub(crate) fn between(
        donor: SpatialGpuUniforms,
        recipient: SpatialGpuUniforms,
    ) -> Option<Self> {
        if donor.modes[2] == 0 || recipient.modes[2] == 0 {
            return None;
        }
        let donor_inverse = [
            [donor.inverse_row_0[0], donor.inverse_row_0[1]],
            [donor.inverse_row_1[0], donor.inverse_row_1[1]],
        ];
        let donor_forward = invert_2x2(donor_inverse)?;
        let recipient_inverse = [
            [recipient.inverse_row_0[0], recipient.inverse_row_0[1]],
            [recipient.inverse_row_1[0], recipient.inverse_row_1[1]],
        ];
        let vector_map = multiply_2x2(recipient_inverse, donor_forward);
        if vector_map
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return None;
        }
        Some(Self {
            output_to_donor_row_0: donor.inverse_row_0,
            output_to_donor_row_1: donor.inverse_row_1,
            donor_to_recipient_row_0: [vector_map[0][0], vector_map[0][1], 0.0, 0.0],
            donor_to_recipient_row_1: [vector_map[1][0], vector_map[1][1], 0.0, 0.0],
        })
    }

    #[cfg(test)]
    fn map_vector(self, vector: [f32; 2]) -> [f32; 2] {
        [
            self.donor_to_recipient_row_0[0] * vector[0]
                + self.donor_to_recipient_row_0[1] * vector[1],
            self.donor_to_recipient_row_1[0] * vector[0]
                + self.donor_to_recipient_row_1[1] * vector[1],
        ]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "motion runtime metrics and CPU laws consume this snapshot"
    )
)]
pub(crate) struct MotionMemoryMetrics {
    pub luma_valid: bool,
    pub field_valid: bool,
    pub carrier_valid: bool,
    pub frame_staged: bool,
    pub committed_field_index: u8,
    pub committed_luma_index: u8,
    pub committed_carrier_index: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MotionMemorySnapshot {
    luma_valid: bool,
    field_valid: bool,
    carrier_valid: bool,
    field_index: u8,
    luma_index: u8,
    carrier_index: u8,
    cadence_accumulator: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct MotionMemoryState {
    committed: MotionMemorySnapshot,
    staged: Option<MotionMemorySnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MotionMemoryStage {
    pub read_field_index: u8,
    pub write_field_index: u8,
    pub render_field_index: u8,
    pub read_luma_index: u8,
    pub write_luma_index: u8,
    pub read_carrier_index: u8,
    pub write_carrier_index: u8,
    pub update_lattice: bool,
    pub field_read_valid: bool,
    pub render_field_valid: bool,
    pub luma_read_valid: bool,
    pub carrier_read_valid: bool,
}

impl MotionMemoryState {
    /// Stages only CPU publication state. GPU callers must write field/gate
    /// data to `write_field_index`, render the next carrier into host scratch,
    /// and encode scratch-to-carrier last. `commit` is called only after that
    /// command buffer has been accepted; abandoning it leaves all committed
    /// resources and this snapshot untouched.
    #[allow(
        clippy::too_many_arguments,
        reason = "independent transport/acquisition/publication gates are intentionally explicit"
    )]
    pub(crate) fn stage(
        &mut self,
        delta_seconds: f32,
        update_hz: u32,
        program_advances: bool,
        field_acquisition_advances: bool,
        lattice_acquisition: bool,
        codec_upload_available: bool,
        persistent_carrier: bool,
    ) -> MotionMemoryStage {
        assert!(self.staged.is_none(), "motion frame already staged");
        let before = self.committed;
        let mut after = before;
        let write_field_index = before.field_index ^ 1;
        let write_luma_index = before.luma_index ^ 1;
        let write_carrier_index = before.carrier_index ^ 1;
        let mut update_lattice = false;
        if program_advances && field_acquisition_advances && lattice_acquisition {
            let delta = if delta_seconds.is_finite() {
                f64::from(delta_seconds.max(0.0))
            } else {
                0.0
            };
            after.cadence_accumulator += delta;
            let interval = 1.0 / f64::from(update_hz.clamp(1, 60));
            if !before.luma_valid || after.cadence_accumulator + f64::EPSILON >= interval {
                update_lattice = true;
                if before.luma_valid {
                    after.cadence_accumulator %= interval;
                } else {
                    after.cadence_accumulator = 0.0;
                }
                after.luma_index = write_luma_index;
                after.luma_valid = true;
                if before.luma_valid {
                    after.field_index = write_field_index;
                    after.field_valid = true;
                }
            }
        }
        if program_advances && field_acquisition_advances && codec_upload_available {
            after.field_index = write_field_index;
            after.field_valid = true;
        }
        // Media Freeze and a paused layer hold acquisition, but their program
        // memory continues against the last committed field/held source.
        if program_advances && persistent_carrier {
            after.carrier_valid = true;
            after.carrier_index = write_carrier_index;
        }
        // Program Freeze is a literal motion-memory hold: no cadence, field,
        // codec product, carrier publication, or staged CPU state exists.
        if program_advances {
            self.staged = Some(after);
        }
        MotionMemoryStage {
            read_field_index: before.field_index,
            write_field_index,
            render_field_index: after.field_index,
            read_luma_index: before.luma_index,
            write_luma_index,
            read_carrier_index: before.carrier_index,
            write_carrier_index,
            update_lattice,
            field_read_valid: before.field_valid,
            render_field_valid: after.field_valid,
            luma_read_valid: before.luma_valid,
            carrier_read_valid: before.carrier_valid,
        }
    }

    pub(crate) fn commit(&mut self) {
        if let Some(after) = self.staged.take() {
            self.committed = after;
        }
    }

    pub(crate) fn discard(&mut self) {
        self.staged = None;
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "Main telemetry consumes the aggregate motion snapshot"
        )
    )]
    pub(crate) fn metrics(self) -> MotionMemoryMetrics {
        MotionMemoryMetrics {
            luma_valid: self.committed.luma_valid,
            field_valid: self.committed.field_valid,
            carrier_valid: self.committed.carrier_valid,
            frame_staged: self.staged.is_some(),
            committed_field_index: self.committed.field_index,
            committed_luma_index: self.committed.luma_index,
            committed_carrier_index: self.committed.carrier_index,
        }
    }
}

/// Visible, fixed resource cost for one curved-shutter/Faraday pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "fixed quality/sample laws are exercised by CPU fixtures"
    )
)]
pub(crate) struct MotionPassBudget {
    pub full_frame_passes: u8,
    pub texture_samples_per_pixel: u8,
    pub max_sampled_textures_in_pass: u8,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "fixed quality/sample laws are exercised by CPU fixtures"
    )
)]
pub(crate) fn motion_pass_budget(
    shutter_samples: u8,
    chromatic_lag: bool,
    faraday_active: bool,
) -> MotionPassBudget {
    let shutter_samples = shutter_samples.clamp(1, 16);
    let carrier_lookups = if chromatic_lag {
        shutter_samples.saturating_mul(3)
    } else {
        shutter_samples
    };
    let carrier_texture_ops = carrier_lookups.saturating_mul(PREMULTIPLIED_BILINEAR_TEXTURE_OPS);
    MotionPassBudget {
        full_frame_passes: 1 + u8::from(faraday_active),
        texture_samples_per_pixel: carrier_texture_ops
            .saturating_add(2)
            // Faraday refresh owns three lookups. Luma acquisition now owns
            // four explicit covered-color loads at its downsample boundary.
            .saturating_add(u8::from(faraday_active) * 7),
        max_sampled_textures_in_pass: 3,
    }
}

fn invert_2x2(value: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let determinant = value[0][0].mul_add(value[1][1], -value[0][1] * value[1][0]);
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse = [
        [value[1][1] / determinant, -value[0][1] / determinant],
        [-value[1][0] / determinant, value[0][0] / determinant],
    ];
    inverse
        .into_iter()
        .flatten()
        .all(|component| component.is_finite())
        .then_some(inverse)
}

fn multiply_2x2(a: [[f32; 2]; 2], b: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    [
        [
            a[0][0].mul_add(b[0][0], a[0][1] * b[1][0]),
            a[0][0].mul_add(b[0][1], a[0][1] * b[1][1]),
        ],
        [
            a[1][0].mul_add(b[0][0], a[1][1] * b[1][0]),
            a[1][0].mul_add(b[0][1], a[1][1] * b[1][1]),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        motion::{
            deterministic_motion_lattice, CurvedShutterParams, LumaPlane, MotionDeviceLimits,
            MotionField, MotionFieldOrigin, MotionGrid, MotionLatticeQuality, MotionParams,
            MotionResourcePlan, MotionScopeResourceRequest, MotionVectorSample,
            MOTION_FIELD_MAX_BYTES,
        },
        spatial::SpatialTransform,
    };

    #[test]
    fn gpu_adapter_consumes_the_canonical_grid_and_budget() {
        let grid = MotionGrid::for_source([7_680, 4_320], MotionLatticeQuality::High).unwrap();
        assert_eq!([grid.width, grid.height], [1_920, 1_080]);
        let params = MotionParams {
            lattice_quality: MotionLatticeQuality::High,
            shutter: CurvedShutterParams {
                angle_degrees: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = MotionResourcePlan::preflight(
            &[MotionScopeResourceRequest {
                source_dimensions: [7_680, 4_320],
                output_dimensions: [7_680, 4_320],
                params,
                is_master: true,
                codec_vectors_available: false,
                required_as_donor: false,
                required_as_garden_signal: false,
            }],
            MotionDeviceLimits::new(8_192, u64::MAX),
        )
        .unwrap();
        assert_eq!(plan.vector_bytes, grid.vector_count * 8);
        assert_eq!(plan.gate_bytes, grid.vector_count * 4);
        assert_eq!(plan.luma_bytes, grid.vector_count * 2);
        assert!(plan.vector_bytes + plan.gate_bytes + plan.luma_bytes <= MOTION_FIELD_MAX_BYTES);
    }

    #[test]
    fn canonical_static_and_known_vectors_pack_for_rg16float_without_a_second_law() {
        let grid = MotionGrid {
            width: 4,
            height: 1,
            block_pixels: 1,
            vector_count: 4,
        };
        let field = MotionField::from_samples(
            [4, 1],
            grid,
            MotionFieldOrigin::CodecVectors,
            [0.0_f32, 1.0, 2.0, 4.0].map(|velocity| MotionVectorSample {
                velocity_uv_per_second: [velocity, -velocity],
                confidence: 0.75,
                visibility: 0.5,
            }),
        )
        .unwrap();
        let mut vectors = [0_u8; 16];
        let mut gates = [0_u8; 8];
        encode_field_upload(&field, &mut vectors, &mut gates).unwrap();
        for (index, expected) in [0x0000_u16, 0x3c00, 0x4000, 0x4400].into_iter().enumerate() {
            let offset = index * 4;
            assert_eq!(
                u16::from_le_bytes([vectors[offset], vectors[offset + 1]]),
                expected
            );
            let expected_y = if expected == 0 { 0 } else { expected | 0x8000 };
            assert_eq!(
                u16::from_le_bytes([vectors[offset + 2], vectors[offset + 3]]),
                expected_y
            );
            assert_eq!(&gates[index * 2..index * 2 + 2], &[191, 128]);
        }
        assert!(matches!(
            encode_field_upload(&field, &mut vectors[..12], &mut gates),
            Err(MotionGpuError::ResourcePlanMismatch)
        ));
    }

    #[test]
    fn rg16float_conversion_is_finite_stable_and_rounds_ties_even() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NAN), 0x7e00);
    }

    #[test]
    fn donor_vectors_transform_through_composition_into_recipient_local_space() {
        let donor = SpatialTransform {
            scale: [2.0, 1.0],
            ..SpatialTransform::default()
        }
        .gpu_uniforms(1_920, 1_080, 1_920, 1_080);
        let recipient = SpatialTransform {
            scale: [1.0, 2.0],
            ..SpatialTransform::default()
        }
        .gpu_uniforms(1_920, 1_080, 1_920, 1_080);
        let transform = MotionTransformGpu::between(donor, recipient).unwrap();
        assert_eq!(transform.map_vector([0.25, 0.5]), [0.5, 0.25]);

        let invalid = SpatialGpuUniforms {
            modes: [0, 0, 0, 1],
            ..donor
        };
        assert_eq!(MotionTransformGpu::between(invalid, recipient), None);
    }

    fn map_spatial_sample(sample: MotionSpatialSampleGpu, uv: [f32; 2]) -> [f32; 2] {
        [
            sample.green_row_0[0].mul_add(
                uv[0],
                sample.green_row_0[1].mul_add(uv[1], sample.green_row_0[2]),
            ),
            sample.green_row_1[0].mul_add(
                uv[0],
                sample.green_row_1[1].mul_add(uv[1], sample.green_row_1[2]),
            ),
        ]
    }

    fn close(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn authored_translation_rotation_and_scale_have_rate_stable_subframe_trajectories() {
        let shutter = CurvedShutterParams {
            angle_degrees: 360.0,
            phase: -1.0,
            ..Default::default()
        };
        let uv = [0.75, 0.5];
        for fps in [24.0_f32, 30.0, 60.0] {
            let translation = SpatialTransform {
                position: [0.3 / fps, -0.15 / fps],
                ..SpatialTransform::default()
            };
            let translated = map_spatial_sample(
                motion_spatial_samples(
                    SpatialTransform::default(),
                    translation,
                    [1_024, 1_024],
                    [1_024, 1_024],
                    shutter,
                )
                .unwrap()[0],
                uv,
            );
            close((translated[0] - uv[0]) * fps, 0.3, 2.0e-5);
            close((translated[1] - uv[1]) * fps, -0.15, 2.0e-5);

            let angle = 90.0_f32 / fps;
            let rotation = SpatialTransform {
                rotation_deg: angle,
                ..SpatialTransform::default()
            };
            let rotated = map_spatial_sample(
                motion_spatial_samples(
                    SpatialTransform::default(),
                    rotation,
                    [1_024, 1_024],
                    [1_024, 1_024],
                    shutter,
                )
                .unwrap()[0],
                uv,
            );
            let radians = angle.to_radians();
            close(rotated[0], 0.5 + 0.25 * radians.cos(), 2.0e-5);
            close(rotated[1], 0.5 + 0.25 * radians.sin(), 2.0e-5);

            let scale_per_frame = (0.6 / fps).exp();
            let scaled = map_spatial_sample(
                motion_spatial_samples(
                    SpatialTransform::default(),
                    SpatialTransform {
                        scale: [scale_per_frame; 2],
                        ..SpatialTransform::default()
                    },
                    [1_024, 1_024],
                    [1_024, 1_024],
                    shutter,
                )
                .unwrap()[0],
                uv,
            );
            close(scaled[0], 0.5 + 0.25 * scale_per_frame, 2.0e-5);
            close(scaled[1], 0.5, 2.0e-5);
        }
    }

    #[test]
    fn authored_spatial_history_is_transactional_and_resettable() {
        let first = SpatialTransform {
            position: [0.1, 0.0],
            ..SpatialTransform::default()
        };
        let rejected = SpatialTransform {
            position: [0.8, 0.0],
            ..SpatialTransform::default()
        };
        let mut memory = MotionSpatialMemory::default();
        let cold = memory.stage(first, true);
        assert_eq!(cold.previous, first);
        memory.commit();

        let staged = memory.stage(rejected, true);
        assert_eq!(staged.previous, first);
        memory.discard();
        let frozen = memory.stage(rejected, false);
        assert_eq!(frozen.previous, first);
        assert!(memory.staged.is_none(), "Program Freeze must not stage");

        memory.reset();
        let reset = memory.stage(rejected, true);
        assert_eq!(reset.previous, rejected);
    }

    #[test]
    fn field_and_carrier_publication_commit_and_discard_transactionally() {
        let mut state = MotionMemoryState::default();
        let prime = state.stage(1.0 / 30.0, 30, true, true, true, false, true);
        assert!(prime.update_lattice);
        assert!(!prime.luma_read_valid);
        assert!(!prime.carrier_read_valid);
        assert_eq!(
            state.metrics(),
            MotionMemoryMetrics {
                frame_staged: true,
                ..Default::default()
            }
        );
        state.discard();
        assert_eq!(state.metrics(), MotionMemoryMetrics::default());

        state.stage(1.0 / 30.0, 30, true, true, true, false, true);
        state.commit();
        let primed = state.metrics();
        assert!(primed.luma_valid);
        assert!(primed.carrier_valid);
        assert!(!primed.field_valid);

        let field = state.stage(1.0 / 30.0, 30, true, true, true, false, true);
        assert!(field.update_lattice);
        assert!(field.luma_read_valid);
        state.commit();
        assert!(state.metrics().field_valid);

        let committed = state.metrics();
        state.stage(1.0 / 30.0, 30, true, true, true, true, true);
        state.discard();
        assert_eq!(state.metrics(), committed);
        state.reset();
        assert_eq!(state.metrics(), MotionMemoryMetrics::default());
    }

    #[test]
    fn freeze_blackout_and_layer_pause_obey_one_motion_clock_law() {
        let mut state = MotionMemoryState::default();
        state.stage(1.0 / 30.0, 30, true, true, true, true, true);
        state.commit();
        let committed = state.metrics();

        // Program Freeze stages no cadence, acquisition, codec publication,
        // or carrier advance even if a codec product was offered.
        let program_freeze = state.stage(10.0, 60, false, true, true, true, true);
        assert!(!program_freeze.update_lattice);
        state.commit();
        assert_eq!(state.metrics(), committed);

        // Media Freeze and a paused layer both hold acquisition while program
        // time can advance the carrier using the committed field/held pixels.
        for _ in 0..2 {
            let held = state.stage(10.0, 60, true, false, true, true, true);
            assert!(!held.update_lattice);
            assert!(held.carrier_read_valid);
            state.commit();
            assert_eq!(
                state.metrics().committed_field_index,
                committed.committed_field_index
            );
            assert_eq!(
                state.metrics().committed_luma_index,
                committed.committed_luma_index
            );
        }

        // Blackout changes presentation only; hidden program motion evolves.
        let blackout = state.stage(1.0 / 60.0, 60, true, true, true, false, true);
        assert!(blackout.update_lattice);
        state.commit();
        assert_ne!(
            state.metrics().committed_luma_index,
            committed.committed_luma_index
        );
    }

    #[test]
    fn fixed_lattice_cadence_is_deterministic_at_24_30_and_60() {
        for (fps, expected) in [(24, [15, 24, 24]), (30, [15, 30, 30]), (60, [15, 30, 60])] {
            let replay = || {
                [15_u32, 30, 60].map(|update_hz| {
                    let mut state = MotionMemoryState::default();
                    let mut updates = 0;
                    for _ in 0..fps {
                        let stage = state.stage(
                            1.0 / fps as f32,
                            update_hz,
                            true,
                            true,
                            true,
                            false,
                            false,
                        );
                        updates += u32::from(stage.update_lattice);
                        state.commit();
                    }
                    updates
                })
            };
            assert_eq!(replay(), expected, "fps={fps}");
            assert_eq!(replay(), expected, "replay fps={fps}");
        }
    }

    #[test]
    fn fixed_shutter_quality_and_chromatic_costs_are_visible_and_bounded() {
        assert_eq!(
            motion_pass_budget(1, false, false),
            MotionPassBudget {
                full_frame_passes: 1,
                texture_samples_per_pixel: 6,
                max_sampled_textures_in_pass: 3,
            }
        );
        assert_eq!(
            motion_pass_budget(16, true, true),
            MotionPassBudget {
                full_frame_passes: 2,
                texture_samples_per_pixel: 201,
                max_sampled_textures_in_pass: 3,
            }
        );
    }

    #[test]
    fn advanced_motion_shaders_use_covered_filtering_and_premultiplied_accumulation() {
        let apply = include_str!("../shaders/motion_apply.wgsl");
        let refresh = include_str!("../shaders/motion_refresh.wgsl");
        let luma = include_str!("../shaders/motion_luma.wgsl");
        let lattice = include_str!("../shaders/motion_lattice.wgsl");
        let garden_signal = include_str!("../shaders/motion_garden_signal.wgsl");
        assert_eq!(apply.matches("textureLoad(carrier_texture").count(), 4);
        assert!(apply.contains("sample.rgb * clamp(sample.a"));
        assert!(apply.contains("straight_from_premultiplied_filter(accumulated"));
        assert!(refresh.contains("premultiply_refresh(textureSample(current_texture"));
        assert!(refresh.contains("premultiply_refresh(textureSample(advected_texture"));
        assert!(
            refresh.contains("straight_from_refresh_premultiplied(mix(current, memory, amount))")
        );
        assert_eq!(luma.matches("textureLoad(source_texture").count(), 4);
        assert!(luma.contains("return dot(covered.rgb"));
        let zero_first = lattice.find("var best_offset = vec2<i32>(0)").unwrap();
        let ring_search = lattice.find("for (var ring = 1").unwrap();
        assert!(zero_first < ring_search);
        assert!(lattice.contains("if (cost < best_cost)"));
        assert!(!lattice.contains("if (cost <= best_cost)"));
        assert!(lattice.contains("let displaced = uv - vec2<f32>(offset) / source_dimensions"));
        assert!(garden_signal.contains("length(velocity) * gate.x * gate.y"));
        assert!(garden_signal.contains("@group(0) @binding(0) var vectors"));
        assert!(garden_signal.contains("@group(0) @binding(1) var gates"));
        assert!(!garden_signal.contains("@binding(3)"));
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MotionWarmAllocationFingerprint {
        plan: MotionResourcePlan,
        field_count: usize,
        field_capacity: usize,
        vector_upload: (usize, usize, usize),
        gate_upload: (usize, usize, usize),
        vector_textures: [usize; 2],
        gate_textures: [usize; 2],
        luma_textures: [usize; 2],
        bindings: usize,
    }

    struct MotionLatticeGpuHarness {
        resources: MotionGpuResources,
        source: MotionTexture,
        dimensions: [u32; 2],
        grid: MotionGrid,
    }

    impl MotionLatticeGpuHarness {
        fn new(device: &wgpu::Device, queue: &wgpu::Queue, dimensions: [u32; 2]) -> Self {
            let quality = MotionLatticeQuality::High;
            let params = MotionParams {
                lattice_quality: quality,
                shutter: CurvedShutterParams {
                    angle_degrees: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let plan = MotionResourcePlan::preflight(
                &[MotionScopeResourceRequest {
                    source_dimensions: dimensions,
                    output_dimensions: dimensions,
                    params,
                    is_master: true,
                    codec_vectors_available: false,
                    required_as_donor: false,
                    required_as_garden_signal: false,
                }],
                MotionDeviceLimits::new(device.limits().max_texture_dimension_2d, u64::MAX),
            )
            .unwrap();
            let grid = MotionGrid::for_source(dimensions, quality).unwrap();
            let mut resources = MotionGpuResources::prepare(
                device,
                plan,
                &[MotionGpuFieldSpec {
                    grid,
                    requires_luma: true,
                    required_as_garden_signal: false,
                }],
                dimensions,
            )
            .unwrap()
            .unwrap();
            let source = MotionTexture::new(
                device,
                "Motion lattice canonical source fixture",
                dimensions,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let uniform = fixed_uniform_buffer(
                device,
                queue,
                "Motion lattice canonical fixture uniform",
                &LatticeGpuUniforms {
                    field_size: [grid.width, grid.height],
                    source_size: dimensions,
                    search_radius: quality.search_radius() as u32,
                    update_hz: quality.update_hz(),
                    algorithm_version: u32::from(MOTION_ALGORITHM_VERSION),
                    _reserved: 0,
                },
            );
            let field = &mut resources.fields[0];
            let luma = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Motion lattice canonical fixture luma BG"),
                layout: &resources.pipelines.luma_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &resources.pipelines.linear_sampler,
                        ),
                    },
                ],
            });
            let luma_views = field.luma.as_ref().unwrap();
            let lattice = std::array::from_fn(|read_index| {
                let write_index = 1 - read_index;
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Motion lattice canonical fixture parity BG"),
                    layout: &resources.pipelines.lattice_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &luma_views[write_index].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &luma_views[read_index].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &resources.pipelines.linear_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                })
            });
            field.bindings = Some(MotionFieldBindings { luma, lattice });
            Self {
                resources,
                source,
                dimensions,
                grid,
            }
        }

        fn allocation_fingerprint(&self) -> MotionWarmAllocationFingerprint {
            let field = &self.resources.fields[0];
            let luma = field.luma.as_ref().unwrap();
            MotionWarmAllocationFingerprint {
                plan: self.resources.plan(),
                field_count: self.resources.fields.len(),
                field_capacity: self.resources.fields.capacity(),
                vector_upload: (
                    field.vector_upload.as_ptr() as usize,
                    field.vector_upload.len(),
                    field.vector_upload.capacity(),
                ),
                gate_upload: (
                    field.gate_upload.as_ptr() as usize,
                    field.gate_upload.len(),
                    field.gate_upload.capacity(),
                ),
                vector_textures: std::array::from_fn(|index| {
                    std::ptr::from_ref(&field.vectors[index].texture) as usize
                }),
                gate_textures: std::array::from_fn(|index| {
                    std::ptr::from_ref(&field.gates[index].texture) as usize
                }),
                luma_textures: std::array::from_fn(|index| {
                    std::ptr::from_ref(&luma[index].texture) as usize
                }),
                bindings: std::ptr::from_ref(field.bindings.as_ref().unwrap()) as usize,
            }
        }

        fn write_source(&self, queue: &wgpu::Queue, luma: &[u8]) {
            assert_eq!(
                luma.len(),
                (self.dimensions[0] * self.dimensions[1]) as usize
            );
            let rgba = luma
                .iter()
                .flat_map(|value| [*value, *value, *value, 255])
                .collect::<Vec<_>>();
            queue.write_texture(
                self.source.texture.as_image_copy(),
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.dimensions[0] * 4),
                    rows_per_image: Some(self.dimensions[1]),
                },
                wgpu::Extent3d {
                    width: self.dimensions[0],
                    height: self.dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
        }

        fn encode_observation(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            source: &[u8],
        ) {
            self.write_source(queue, source);
            let field = &mut self.resources.fields[0];
            let stage = field.memory.stage(
                1.0 / MotionLatticeQuality::High.update_hz() as f32,
                MotionLatticeQuality::High.update_hz(),
                true,
                true,
                true,
                false,
                false,
            );
            assert!(stage.update_lattice);
            field.frame_stage = Some(stage);
            let bindings = field.bindings.as_ref().unwrap();
            let luma = field.luma.as_ref().unwrap();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Motion lattice canonical fixture encoder"),
            });
            encode_single_target_pass(
                &mut encoder,
                "Motion lattice canonical fixture luma",
                &self.resources.pipelines.luma,
                &bindings.luma,
                &luma[usize::from(stage.write_luma_index)].view,
                None,
            );
            if stage.luma_read_valid {
                encode_lattice_pass(
                    &mut encoder,
                    &self.resources.pipelines.lattice,
                    &bindings.lattice[usize::from(stage.read_luma_index)],
                    &field.vectors[usize::from(stage.write_field_index)].view,
                    &field.gates[usize::from(stage.write_field_index)].view,
                );
            }
            queue.submit(Some(encoder.finish()));
            self.resources.commit_frame();
        }

        fn run_fixture(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            previous: &[u8],
            current: &[u8],
        ) -> Vec<u8> {
            self.resources.reset();
            self.encode_observation(device, queue, previous);
            self.encode_observation(device, queue, current);
            let metrics = self.resources.fields[0].memory.metrics();
            assert!(metrics.field_valid);
            read_motion_vectors(
                device,
                queue,
                &self.resources.fields[0].vectors[usize::from(metrics.committed_field_index)]
                    .texture,
                self.grid,
            )
        }
    }

    fn read_motion_vectors(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        grid: MotionGrid,
    ) -> Vec<u8> {
        let compact_row_bytes = grid.width * 4;
        let padded_row_bytes = compact_row_bytes.next_multiple_of(256);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Motion lattice canonical vector readback"),
            size: u64::from(padded_row_bytes) * u64::from(grid.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Motion lattice canonical vector readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(grid.height),
                },
            },
            wgpu::Extent3d {
                width: grid.width,
                height: grid.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("motion lattice vector readback wait");
        receive.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let compact_row_bytes = compact_row_bytes as usize;
        let padded_row_bytes = padded_row_bytes as usize;
        let mut compact = Vec::with_capacity(compact_row_bytes * grid.height as usize);
        for row in mapped.chunks_exact(padded_row_bytes) {
            compact.extend_from_slice(&row[..compact_row_bytes]);
        }
        drop(mapped);
        buffer.unmap();
        compact
    }

    fn canonical_luma_fixture(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .map(|index| {
                let x = (index % width) as f32;
                let y = (index / width) as f32;
                let signal = 0.5
                    + 0.18 * (x * 0.13).sin()
                    + 0.17 * (y * 0.17).cos()
                    + 0.12 * ((x + y) * 0.09).sin();
                (signal.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    fn shifted_luma_fixture(previous: &[u8], width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
        let mut current = vec![0; previous.len()];
        for y in 0..height {
            for x in 0..width {
                let source_x = i64::from(x) - i64::from(dx);
                let source_y = i64::from(y) - i64::from(dy);
                if source_x >= 0
                    && source_y >= 0
                    && source_x < i64::from(width)
                    && source_y < i64::from(height)
                {
                    current[(y * width + x) as usize] = previous[(u32::try_from(source_y).unwrap()
                        * width
                        + u32::try_from(source_x).unwrap())
                        as usize];
                }
            }
        }
        current
    }

    fn canonical_cpu_vectors(previous: &[u8], current: &[u8], dimensions: [u32; 2]) -> Vec<u8> {
        let plane = |pixels| LumaPlane {
            width: dimensions[0],
            height: dimensions[1],
            stride: dimensions[0] as usize,
            pixels,
        };
        let field = deterministic_motion_lattice(
            plane(previous),
            plane(current),
            MotionLatticeQuality::High,
        )
        .unwrap();
        let mut vectors = vec![0; field.packed_vectors().len() * 4];
        let mut gates = vec![0; field.packed_vectors().len() * 2];
        encode_field_upload(&field, &mut vectors, &mut gates).unwrap();
        vectors
    }

    fn motion_half_to_f32(value: u16) -> f32 {
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

    #[test]
    #[ignore = "requires an opted-in physical GPU readback adapter"]
    fn gpu_motion_lattice_matches_cpu_known_shifts_static_zero_live_export_and_warm_law() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("motion lattice physical GPU adapter");
        let adapter_info = adapter.get_info();
        println!(
            "motion lattice physical adapter: {} ({:?}, {:?})",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Motion lattice physical acceptance device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("motion lattice physical GPU device");
        let dimensions = [64, 48];
        let previous = canonical_luma_fixture(dimensions[0], dimensions[1]);
        let uniform = vec![127; previous.len()];
        let fixtures = [
            (
                "one-pixel",
                previous.clone(),
                shifted_luma_fixture(&previous, 64, 48, 1, 0),
                1,
                0,
            ),
            (
                "two-pixel",
                previous.clone(),
                shifted_luma_fixture(&previous, 64, 48, 0, 2),
                0,
                2,
            ),
            (
                "four-pixel",
                previous.clone(),
                shifted_luma_fixture(&previous, 64, 48, 4, -4),
                4,
                -4,
            ),
            ("static-uniform", uniform.clone(), uniform, 0, 0),
        ];
        let mut live = MotionLatticeGpuHarness::new(&device, &queue, dimensions);
        let mut export = MotionLatticeGpuHarness::new(&device, &queue, dimensions);
        let live_allocation = live.allocation_fingerprint();
        let export_allocation = export.allocation_fingerprint();
        assert_eq!(live_allocation.plan, export_allocation.plan);
        assert_eq!(live_allocation.plan.active_field_slots, 1);
        assert_eq!(live_allocation.plan.vector_bytes, 16 * 12 * 8);
        assert_eq!(live_allocation.plan.gate_bytes, 16 * 12 * 4);
        assert_eq!(live_allocation.plan.luma_bytes, 16 * 12 * 2);

        for (label, previous, current, dx, dy) in fixtures {
            let expected = canonical_cpu_vectors(&previous, &current, dimensions);
            let live_vectors = live.run_fixture(&device, &queue, &previous, &current);
            let export_vectors = export.run_fixture(&device, &queue, &previous, &current);
            assert_eq!(live_vectors, export_vectors, "{label} live/export parity");
            let center = ((6 * 16 + 8) * 4) as usize;
            for (vectors, path) in [(&expected, "CPU oracle"), (&live_vectors, "GPU renderer")] {
                let velocity = [
                    motion_half_to_f32(u16::from_le_bytes([vectors[center], vectors[center + 1]])),
                    motion_half_to_f32(u16::from_le_bytes([
                        vectors[center + 2],
                        vectors[center + 3],
                    ])),
                ];
                let recovered = [velocity[0] * 64.0 / 60.0, velocity[1] * 48.0 / 60.0];
                assert!(
                    (recovered[0] - dx as f32).abs() < 0.01
                        && (recovered[1] - dy as f32).abs() < 0.01,
                    "{label} {path} recovered {recovered:?}, expected [{dx}, {dy}]"
                );
            }
            if label == "static-uniform" {
                assert!(
                    live_vectors.iter().all(|byte| *byte == 0),
                    "static uniform field must publish literal +0 RG16F vectors"
                );
            }
        }
        assert_eq!(live.allocation_fingerprint(), live_allocation);
        assert_eq!(export.allocation_fingerprint(), export_allocation);
        assert_eq!(live.resources.memory_generation, 2);
        assert_eq!(export.resources.memory_generation, 2);
    }

    #[test]
    fn gpu_motion_formats_pipelines_and_codec_upload_are_valid_when_opted_in() {
        if std::env::var_os("COLLIDE_GPU_GOLDENS").is_none() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("motion GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Motion GPU fixture"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("motion GPU device");
        let dimensions = [64, 32];
        let params = MotionParams {
            shutter: CurvedShutterParams {
                angle_degrees: 30.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let request = MotionScopeResourceRequest {
            source_dimensions: dimensions,
            output_dimensions: dimensions,
            params,
            is_master: true,
            codec_vectors_available: false,
            required_as_donor: false,
            required_as_garden_signal: true,
        };
        let plan = MotionResourcePlan::preflight(
            &[request],
            MotionDeviceLimits::new(device.limits().max_texture_dimension_2d, u64::MAX),
        )
        .unwrap();
        let grid = MotionGrid::for_source(dimensions, params.lattice_quality).unwrap();
        let mut resources = MotionGpuResources::prepare(
            &device,
            plan,
            &[MotionGpuFieldSpec {
                grid,
                requires_luma: true,
                required_as_garden_signal: true,
            }],
            dimensions,
        )
        .unwrap()
        .unwrap();
        let field = MotionField::from_samples(
            dimensions,
            grid,
            MotionFieldOrigin::CodecVectors,
            std::iter::repeat_n(
                MotionVectorSample {
                    velocity_uv_per_second: [2.0, -4.0],
                    confidence: 1.0,
                    visibility: 1.0,
                },
                usize::try_from(grid.vector_count).unwrap(),
            ),
        )
        .unwrap();
        resources.upload_codec_field(&queue, 0, &field, 1).unwrap();
        assert_eq!(resources.plan(), plan);
        assert!(resources.garden_signal_view().is_some());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("motion fixture wait");
    }

    #[test]
    fn submitted_then_discarded_carrier_keeps_committed_gpu_parity_when_opted_in() {
        if std::env::var_os("COLLIDE_GPU_GOLDENS").is_none() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("motion GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Motion carrier transaction fixture"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("motion GPU device");
        let dimensions = [2, 1];
        let plan = MotionResourcePlan::preflight(
            &[MotionScopeResourceRequest {
                source_dimensions: dimensions,
                output_dimensions: dimensions,
                params: MotionParams {
                    transplant: crate::motion::FaradayParams {
                        amount: 1.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                is_master: false,
                codec_vectors_available: false,
                required_as_donor: false,
                required_as_garden_signal: false,
            }],
            MotionDeviceLimits::new(device.limits().max_texture_dimension_2d, u64::MAX),
        )
        .unwrap();
        assert_eq!(plan.active_field_slots, 0);
        assert_eq!(plan.persistent_carriers, 1);
        let mut resources = MotionGpuResources::prepare(&device, plan, &[], dimensions)
            .unwrap()
            .unwrap();

        // Publish green into parity 1.
        let prime = resources
            .carrier_memory
            .stage(1.0 / 30.0, 1, true, false, false, false, true);
        resources.carrier_stage = Some(prime);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Motion carrier committed fixture encoder"),
        });
        clear_target(
            &mut encoder,
            &resources.carriers.as_ref().unwrap()[usize::from(prime.write_carrier_index)].view,
            wgpu::Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        );
        queue.submit(Some(encoder.finish()));
        resources.commit_frame();
        assert_eq!(resources.memory_generation, 1);
        assert_eq!(
            resources.carrier_memory.metrics().committed_carrier_index,
            1
        );

        // A later submitted red stage targets parity 0, then CPU rejection
        // discards it. The committed green pixels in parity 1 must survive.
        let rejected =
            resources
                .carrier_memory
                .stage(1.0 / 30.0, 1, true, false, false, false, true);
        resources.carrier_stage = Some(rejected);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Motion carrier rejected fixture encoder"),
        });
        clear_target(
            &mut encoder,
            &resources.carriers.as_ref().unwrap()[usize::from(rejected.write_carrier_index)].view,
            wgpu::Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );
        queue.submit(Some(encoder.finish()));
        resources.discard_frame();
        assert_eq!(resources.memory_generation, 1);
        assert_eq!(
            resources.carrier_memory.metrics().committed_carrier_index,
            1
        );

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Motion carrier transaction readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Motion carrier transaction readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            resources.carriers.as_ref().unwrap()[1]
                .texture
                .as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..8);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("motion carrier readback wait");
        receive.recv().expect("map callback").expect("map result");
        let bytes = slice.get_mapped_range();
        assert_eq!(
            &bytes[..8],
            &[0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c]
        );
        drop(bytes);
        staging.unmap();
    }
}
