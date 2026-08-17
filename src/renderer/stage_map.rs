//! Bounded post-creative StageMap GPU presentation.
//!
//! The presenter samples a caller-owned completed Program view and renders only
//! into dedicated endpoint textures. It never owns or writes the creative,
//! audience, Spout, recorder, or export surfaces.

#![allow(
    dead_code,
    reason = "M5 StageMap presenter is consumed by Main after its API freeze"
)]

use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::stage_map::{
    OutputBinding, OutputEndpointId, StageCalibration, StageDeviceLimits, StageEndpointPlan,
    StageEndpointRuntimeError, StageMap, StageMask, StageRoute, StageSurface, StageToolState,
    TestCardMode, MAX_OUTPUT_ENDPOINTS,
};

pub(crate) const STAGE_ENDPOINT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub(crate) const STAGE_ENDPOINT_BYTES_PER_PIXEL: u64 = 4;
pub(crate) const STAGE_PRESENTER_DEFAULT_GPU_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const STAGE_PRESENTER_HARD_MAX_GPU_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const STAGE_PRESENTER_MAX_PROGRAM_SOURCES: usize = 4;
pub(crate) const STAGE_PRESENTER_MAX_SURFACE_FORMATS: usize = 4;
pub(crate) const STAGE_PRESENTER_DIAGNOSTIC_MAX_BYTES: usize = 512;

const STAGE_UNIFORM_BYTES: u64 = std::mem::size_of::<StageGpuUniforms>() as u64;
const STAGE_VERTEX_BYTES: u64 = std::mem::size_of::<StageGpuVertex>() as u64;
const STAGE_INDEX_BYTES: u64 = std::mem::size_of::<u16>() as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StageProgramSourceId(u8);

impl StageProgramSourceId {
    pub(crate) const fn new(value: u8) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u8 {
        self.0
    }
}

pub(crate) struct StageProgramSource<'a> {
    pub id: StageProgramSourceId,
    pub view: &'a wgpu::TextureView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StagePresenterLimits {
    pub stage: StageDeviceLimits,
    pub max_total_gpu_bytes: u64,
    pub max_buffer_size: u64,
    pub uniform_alignment: u64,
    pub max_program_sources: usize,
}

impl StagePresenterLimits {
    pub(crate) fn for_device(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self::bounded(
            StageDeviceLimits {
                max_dimension: limits.max_texture_dimension_2d,
                ..StageDeviceLimits::default()
            },
            STAGE_PRESENTER_DEFAULT_GPU_BYTES,
            limits.max_buffer_size,
            u64::from(limits.min_uniform_buffer_offset_alignment),
            STAGE_PRESENTER_MAX_PROGRAM_SOURCES,
        )
    }

    pub(crate) fn bounded(
        stage: StageDeviceLimits,
        max_total_gpu_bytes: u64,
        max_buffer_size: u64,
        uniform_alignment: u64,
        max_program_sources: usize,
    ) -> Self {
        Self {
            stage,
            max_total_gpu_bytes: max_total_gpu_bytes.min(STAGE_PRESENTER_HARD_MAX_GPU_BYTES),
            max_buffer_size,
            uniform_alignment: uniform_alignment.max(1),
            max_program_sources: max_program_sources.min(STAGE_PRESENTER_MAX_PROGRAM_SOURCES),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StagePresenterResourcePlan {
    pub ready_endpoints: usize,
    pub output_pixels: u64,
    pub output_texture_bytes: u64,
    pub mesh_buffer_bytes: u64,
    pub uniform_buffer_bytes: u64,
    pub total_gpu_bytes: u64,
    pub textures: u64,
    pub views: u64,
    pub buffers: u64,
    pub bind_groups: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StageEndpointPrepareError {
    Domain(StageEndpointRuntimeError),
    ArithmeticOverflow,
    BufferSizeExceeded,
    TotalGpuBudgetExceeded { requested: u64, remaining: u64 },
    ResourceCreation(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StageEndpointPreparationStatus {
    Disabled,
    Ready {
        output_size: [u32; 2],
        refresh_millihz: u32,
        slices: usize,
    },
    Rejected(StageEndpointPrepareError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageEndpointPreparation {
    pub endpoint_id: OutputEndpointId,
    pub binding: OutputBinding,
    pub status: StageEndpointPreparationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StagePresenterPrepareError {
    InvalidStageMap(String),
    InvalidLimits,
    TooManyProgramSources(usize),
    TooManySurfaceFormats(usize),
    DuplicateProgramSource(StageProgramSourceId),
    ResourceCreation(String),
}

impl fmt::Display for StagePresenterPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "StageMap presenter preparation failed: {self:?}")
    }
}

impl std::error::Error for StagePresenterPrepareError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageEndpointFrameStatus {
    NotRendered,
    Presented,
    ProgramSourceUnavailable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StagePresenterFrameMetrics {
    pub presented_endpoints: u32,
    pub program_endpoints: u32,
    pub blackout_endpoints: u32,
    pub test_card_endpoints: u32,
    pub program_source_unavailable: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StageSurfaceFrameMetrics {
    pub presented_surfaces: u32,
    pub missing_endpoints: u32,
    pub unassigned_endpoints: u32,
    pub unsupported_formats: u32,
    pub dimension_mismatches: u32,
    pub duplicate_targets: u32,
    pub excess_targets: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StagePresenterAllocationSnapshot {
    pub textures: u64,
    pub views: u64,
    pub buffers: u64,
    pub bind_groups: u64,
    pub bind_group_layouts: u64,
    pub pipeline_layouts: u64,
    pub pipelines: u64,
    pub shader_modules: u64,
    pub samplers: u64,
    pub texture_bytes: u64,
    pub buffer_bytes: u64,
}

impl StagePresenterAllocationSnapshot {
    pub(crate) const fn total_objects(self) -> u64 {
        self.textures
            + self.views
            + self.buffers
            + self.bind_groups
            + self.bind_group_layouts
            + self.pipeline_layouts
            + self.pipelines
            + self.shader_modules
            + self.samplers
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveRoute {
    Program,
    Blackout,
    TestCard(TestCardMode),
}

impl EffectiveRoute {
    const fn gpu_code(self) -> u32 {
        match self {
            Self::Program => 0,
            Self::Blackout => 1,
            Self::TestCard(TestCardMode::SmpteBars) => 2,
            Self::TestCard(TestCardMode::Grid) => 3,
            Self::TestCard(TestCardMode::Off) => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EndpointResourceEstimate {
    output_pixels: u64,
    output_texture_bytes: u64,
    mesh_buffer_bytes: u64,
    uniform_buffer_bytes: u64,
    total_gpu_bytes: u64,
    vertex_count: usize,
    index_count: usize,
    uniform_stride: u64,
    buffers: u64,
}

enum PlannedEndpointAdmission {
    Disabled,
    Rejected(StageEndpointPrepareError),
    Ready {
        plan: StageEndpointPlan,
        resources: EndpointResourceEstimate,
    },
}

struct PlannedEndpoint {
    id: OutputEndpointId,
    binding: OutputBinding,
    admission: PlannedEndpointAdmission,
}

pub(crate) struct StagePresenterPlan {
    endpoints: Box<[PlannedEndpoint]>,
    resources: StagePresenterResourcePlan,
}

impl StagePresenterPlan {
    pub(crate) fn build(
        stage_map: &StageMap,
        limits: StagePresenterLimits,
        program_source_count: usize,
        endpoint_available: impl FnMut(&crate::stage_map::StageEndpoint) -> Result<(), String>,
    ) -> Result<Self, StagePresenterPrepareError> {
        stage_map.validate().map_err(|error| {
            StagePresenterPrepareError::InvalidStageMap(bounded_diagnostic(error.to_string()))
        })?;
        if limits.max_total_gpu_bytes == 0
            || limits.max_buffer_size < STAGE_UNIFORM_BYTES
            || !limits.uniform_alignment.is_power_of_two()
        {
            return Err(StagePresenterPrepareError::InvalidLimits);
        }
        if program_source_count > limits.max_program_sources {
            return Err(StagePresenterPrepareError::TooManyProgramSources(
                program_source_count,
            ));
        }

        let evaluations = stage_map.evaluate_isolated(limits.stage, endpoint_available);
        let mut resources = StagePresenterResourcePlan {
            bind_groups: program_source_count as u64,
            ..StagePresenterResourcePlan::default()
        };
        let mut endpoints = Vec::with_capacity(evaluations.len());
        for (evaluation, authored) in evaluations.into_iter().zip(&stage_map.endpoints) {
            let id = evaluation.endpoint_id;
            let admission = match evaluation.result {
                Ok(None) => PlannedEndpointAdmission::Disabled,
                Err(error) => PlannedEndpointAdmission::Rejected(
                    StageEndpointPrepareError::Domain(bound_runtime_error(error)),
                ),
                Ok(Some(plan)) => match estimate_endpoint(&plan, limits) {
                    Err(error) => PlannedEndpointAdmission::Rejected(error),
                    Ok(estimate) => {
                        let remaining = limits
                            .max_total_gpu_bytes
                            .saturating_sub(resources.total_gpu_bytes);
                        if estimate.total_gpu_bytes > remaining {
                            PlannedEndpointAdmission::Rejected(
                                StageEndpointPrepareError::TotalGpuBudgetExceeded {
                                    requested: estimate.total_gpu_bytes,
                                    remaining,
                                },
                            )
                        } else {
                            resources.ready_endpoints += 1;
                            resources.output_pixels += estimate.output_pixels;
                            resources.output_texture_bytes += estimate.output_texture_bytes;
                            resources.mesh_buffer_bytes += estimate.mesh_buffer_bytes;
                            resources.uniform_buffer_bytes += estimate.uniform_buffer_bytes;
                            resources.total_gpu_bytes += estimate.total_gpu_bytes;
                            resources.textures += 1;
                            resources.views += 1;
                            resources.buffers += estimate.buffers;
                            resources.bind_groups += 2;
                            PlannedEndpointAdmission::Ready {
                                plan,
                                resources: estimate,
                            }
                        }
                    }
                },
            };
            debug_assert_eq!(id, authored.id);
            endpoints.push(PlannedEndpoint {
                id,
                binding: authored.binding.clone(),
                admission,
            });
        }
        debug_assert!(resources.ready_endpoints <= MAX_OUTPUT_ENDPOINTS);
        if resources.ready_endpoints != 0 {
            resources.bind_groups += 1;
        }
        Ok(Self {
            endpoints: endpoints.into_boxed_slice(),
            resources,
        })
    }

    pub(crate) const fn resources(&self) -> StagePresenterResourcePlan {
        self.resources
    }

    pub(crate) fn preparation(
        &self,
    ) -> impl ExactSizeIterator<Item = StageEndpointPreparation> + '_ {
        self.endpoints
            .iter()
            .map(|endpoint| StageEndpointPreparation {
                endpoint_id: endpoint.id.clone(),
                binding: endpoint.binding.clone(),
                status: match &endpoint.admission {
                    PlannedEndpointAdmission::Disabled => StageEndpointPreparationStatus::Disabled,
                    PlannedEndpointAdmission::Rejected(error) => {
                        StageEndpointPreparationStatus::Rejected(error.clone())
                    }
                    PlannedEndpointAdmission::Ready { plan, .. } => {
                        StageEndpointPreparationStatus::Ready {
                            output_size: plan.output_size,
                            refresh_millihz: plan.refresh_millihz,
                            slices: plan.slices.len(),
                        }
                    }
                },
            })
    }
}

fn estimate_endpoint(
    plan: &StageEndpointPlan,
    limits: StagePresenterLimits,
) -> Result<EndpointResourceEstimate, StageEndpointPrepareError> {
    let output_pixels = u64::from(plan.output_size[0])
        .checked_mul(u64::from(plan.output_size[1]))
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let output_texture_bytes = output_pixels
        .checked_mul(STAGE_ENDPOINT_BYTES_PER_PIXEL)
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let vertex_count = plan.slices.iter().try_fold(0_usize, |total, slice| {
        total
            .checked_add(slice.vertices.len())
            .ok_or(StageEndpointPrepareError::ArithmeticOverflow)
    })?;
    let index_count = plan.slices.iter().try_fold(0_usize, |total, slice| {
        total
            .checked_add(slice.indices.len())
            .ok_or(StageEndpointPrepareError::ArithmeticOverflow)
    })?;
    let vertex_bytes = u64::try_from(vertex_count)
        .ok()
        .and_then(|count| count.checked_mul(STAGE_VERTEX_BYTES))
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let index_bytes = u64::try_from(index_count)
        .ok()
        .and_then(|count| count.checked_mul(STAGE_INDEX_BYTES))
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let mesh_buffer_bytes = vertex_bytes
        .checked_add(index_bytes)
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let uniform_stride = align_up(STAGE_UNIFORM_BYTES, limits.uniform_alignment)
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let uniform_slots = u64::try_from(plan.slices.len() + 1)
        .map_err(|_| StageEndpointPrepareError::ArithmeticOverflow)?;
    let uniform_buffer_bytes = uniform_stride
        .checked_mul(uniform_slots)
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    if vertex_bytes > limits.max_buffer_size
        || index_bytes > limits.max_buffer_size
        || uniform_buffer_bytes > limits.max_buffer_size
        || uniform_buffer_bytes.saturating_sub(uniform_stride) > u64::from(u32::MAX)
    {
        return Err(StageEndpointPrepareError::BufferSizeExceeded);
    }
    let total_gpu_bytes = output_texture_bytes
        .checked_add(mesh_buffer_bytes)
        .and_then(|bytes| bytes.checked_add(uniform_buffer_bytes))
        .ok_or(StageEndpointPrepareError::ArithmeticOverflow)?;
    let buffers = 1 + u64::from(vertex_count != 0) + u64::from(index_count != 0);
    Ok(EndpointResourceEstimate {
        output_pixels,
        output_texture_bytes,
        mesh_buffer_bytes,
        uniform_buffer_bytes,
        total_gpu_bytes,
        vertex_count,
        index_count,
        uniform_stride,
        buffers,
    })
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|rounded| rounded & !(alignment - 1))
}

fn bound_runtime_error(error: StageEndpointRuntimeError) -> StageEndpointRuntimeError {
    match error {
        StageEndpointRuntimeError::Unavailable(message) => {
            StageEndpointRuntimeError::Unavailable(bounded_diagnostic(message))
        }
        error => error,
    }
}

fn bounded_diagnostic(message: String) -> String {
    let mut bounded = String::new();
    for character in message.chars().filter(|character| !character.is_control()) {
        if bounded.len().saturating_add(character.len_utf8()) > STAGE_PRESENTER_DIAGNOSTIC_MAX_BYTES
        {
            break;
        }
        bounded.push(character);
    }
    bounded
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StageGpuVertex {
    source_uv: [f32; 2],
    output_uv: [f32; 2],
}

const STAGE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 8,
        shader_location: 1,
    },
];

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StageGpuUniforms {
    homography_0: [f32; 4],
    homography_1: [f32; 4],
    homography_2: [f32; 4],
    calibration: [f32; 4],
    gain_mask: [f32; 4],
    black_invert: [f32; 4],
    feather: [f32; 4],
    bounds: [f32; 4],
    mask_points_0: [f32; 4],
    mask_points_1: [f32; 4],
    mask_points_2: [f32; 4],
    mask_points_3: [f32; 4],
    modes: [u32; 4],
    surface: [u32; 4],
    reserved_0: [u32; 4],
    reserved_1: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<StageGpuUniforms>() == 256);
const _: () = assert!(std::mem::size_of::<StageGpuVertex>() == 16);

impl StageGpuUniforms {
    fn for_slice(slice: &crate::stage_map::StageSlicePlan) -> Self {
        let homography = slice
            .output_to_source
            .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut points = [[0.0; 2]; 8];
        let (mask_kind, mask_count, feather, invert) = match &slice.mask {
            StageMask::None => (0, 0, [0.0; 4], false),
            StageMask::EdgeFeather { softness } => (1, 0, *softness, false),
            StageMask::Polygon {
                points: authored,
                invert,
                softness,
            } => {
                let mut ordered = authored.clone();
                if signed_polygon_area(&ordered) < 0.0 {
                    ordered.reverse();
                }
                for (target, source) in points.iter_mut().zip(ordered.iter().copied()) {
                    *target = source;
                }
                (2, ordered.len() as u32, [*softness, 0.0, 0.0, 0.0], *invert)
            }
        };
        let bounds = slice.vertices.iter().fold(
            [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            |mut bounds, vertex| {
                bounds[0] = bounds[0].min(vertex.output_uv[0]);
                bounds[1] = bounds[1].min(vertex.output_uv[1]);
                bounds[2] = bounds[2].max(vertex.output_uv[0]);
                bounds[3] = bounds[3].max(vertex.output_uv[1]);
                bounds
            },
        );
        let StageCalibration {
            opacity,
            brightness,
            contrast,
            gamma,
            gain,
            black_level,
        } = slice.calibration;
        Self {
            homography_0: [homography[0], homography[1], homography[2], 0.0],
            homography_1: [homography[3], homography[4], homography[5], 0.0],
            homography_2: [homography[6], homography[7], homography[8], 0.0],
            calibration: [opacity, brightness, contrast, gamma],
            gain_mask: [gain[0], gain[1], gain[2], 0.0],
            black_invert: [
                black_level[0],
                black_level[1],
                black_level[2],
                u32::from(invert) as f32,
            ],
            feather,
            bounds,
            mask_points_0: [points[0][0], points[0][1], points[1][0], points[1][1]],
            mask_points_1: [points[2][0], points[2][1], points[3][0], points[3][1]],
            mask_points_2: [points[4][0], points[4][1], points[5][0], points[5][1]],
            mask_points_3: [points[6][0], points[6][1], points[7][0], points[7][1]],
            modes: [
                u32::from(slice.output_to_source.is_some()),
                mask_kind,
                mask_count,
                0,
            ],
            surface: [0; 4],
            reserved_0: [0; 4],
            reserved_1: [0; 4],
        }
    }

    fn for_surface(route: EffectiveRoute, output_id: bool, endpoint: &OutputEndpointId) -> Self {
        let mut uniforms = Self::zeroed();
        uniforms.surface = [
            route.gpu_code(),
            u32::from(output_id),
            endpoint_hash(endpoint),
            0,
        ];
        uniforms
    }
}

fn signed_polygon_area(points: &[[f32; 2]]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum::<f32>()
        * 0.5
}

fn endpoint_hash(endpoint: &OutputEndpointId) -> u32 {
    endpoint
        .as_str()
        .bytes()
        .fold(2_166_136_261_u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(16_777_619)
        })
}

struct StagePipelines {
    source_layout: wgpu::BindGroupLayout,
    uniform_layout: wgpu::BindGroupLayout,
    empty_bind_group: wgpu::BindGroup,
    slice: wgpu::RenderPipeline,
    surface_replace: wgpu::RenderPipeline,
    surface_overlay: wgpu::RenderPipeline,
    surface_present: Box<[StageSurfacePipeline]>,
    sampler: wgpu::Sampler,
}

struct StageSurfacePipeline {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
}

impl StagePipelines {
    fn build(device: &wgpu::Device, surface_formats: &[wgpu::TextureFormat]) -> Self {
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StageMap Program source layout"),
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
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StageMap dynamic uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(STAGE_UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StageMap empty surface layout"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StageMap empty surface bind group"),
            layout: &empty_layout,
            entries: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("StageMap linear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("StageMap presenter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/stage_map.wgsl").into()),
        });
        let slice_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StageMap slice pipeline layout"),
            bind_group_layouts: &[Some(&source_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let surface_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StageMap surface pipeline layout"),
            bind_group_layouts: &[Some(&empty_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let present_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StageMap physical surface pipeline layout"),
            bind_group_layouts: &[Some(&source_layout)],
            immediate_size: 0,
        });
        let slice = create_stage_pipeline(
            device,
            "StageMap slice pipeline",
            &slice_layout,
            &module,
            "vs_slice",
            "fs_slice",
            &[wgpu::VertexBufferLayout {
                array_stride: STAGE_VERTEX_BYTES,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &STAGE_VERTEX_ATTRIBUTES,
            }],
            STAGE_ENDPOINT_FORMAT,
            Some(premultiplied_alpha_blend()),
        );
        let surface_replace = create_stage_pipeline(
            device,
            "StageMap test-card pipeline",
            &surface_layout,
            &module,
            "vs_surface",
            "fs_surface",
            &[],
            STAGE_ENDPOINT_FORMAT,
            Some(wgpu::BlendState::REPLACE),
        );
        let surface_overlay = create_stage_pipeline(
            device,
            "StageMap output-ID overlay pipeline",
            &surface_layout,
            &module,
            "vs_surface",
            "fs_surface",
            &[],
            STAGE_ENDPOINT_FORMAT,
            Some(premultiplied_alpha_blend()),
        );
        let surface_present = surface_formats
            .iter()
            .copied()
            .map(|format| StageSurfacePipeline {
                format,
                pipeline: create_stage_pipeline(
                    device,
                    "StageMap physical surface blit pipeline",
                    &present_layout,
                    &module,
                    "vs_surface",
                    "fs_present",
                    &[],
                    format,
                    Some(wgpu::BlendState::REPLACE),
                ),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            source_layout,
            uniform_layout,
            empty_bind_group,
            slice,
            surface_replace,
            surface_overlay,
            surface_present,
            sampler,
        }
    }
}

fn premultiplied_alpha_blend() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

#[allow(clippy::too_many_arguments)]
fn create_stage_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    target_format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex_entry),
            buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

struct PreparedProgramSource {
    id: StageProgramSourceId,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone)]
struct StageDraw {
    index_range: Range<u32>,
    base_vertex: i32,
    uniform_offset: u32,
}

struct PreparedStageEndpoint {
    id: OutputEndpointId,
    binding: OutputBinding,
    output_size: [u32; 2],
    refresh_millihz: u32,
    authored_route: StageRoute,
    effective_route: EffectiveRoute,
    output_identification: bool,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    present_bind_group: wgpu::BindGroup,
    draws: Box<[StageDraw]>,
    surface_uniform_offset: u32,
    uniform_stride: u64,
    frame_status: StageEndpointFrameStatus,
    resources: EndpointResourceEstimate,
}

impl PreparedStageEndpoint {
    fn build(
        device: &wgpu::Device,
        pipelines: &StagePipelines,
        plan: StageEndpointPlan,
        binding: OutputBinding,
        resources: EndpointResourceEstimate,
        tools: &StageToolState,
    ) -> Self {
        debug_assert_eq!(
            resources.vertex_count,
            plan.slices.iter().map(|s| s.vertices.len()).sum::<usize>()
        );
        debug_assert_eq!(
            resources.index_count,
            plan.slices.iter().map(|s| s.indices.len()).sum::<usize>()
        );
        let label = format!("StageMap endpoint {}", plan.id);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&label),
            size: wgpu::Extent3d {
                width: plan.output_size[0],
                height: plan.output_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: STAGE_ENDPOINT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let vertices = plan
            .slices
            .iter()
            .flat_map(|slice| {
                slice.vertices.iter().map(|vertex| StageGpuVertex {
                    source_uv: vertex.source_uv,
                    output_uv: vertex.output_uv,
                })
            })
            .collect::<Vec<_>>();
        let indices = plan
            .slices
            .iter()
            .flat_map(|slice| slice.indices.iter().copied())
            .collect::<Vec<_>>();
        let vertex_buffer = (!vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("StageMap endpoint vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let index_buffer = (!indices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("StageMap endpoint indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        });

        let mut uniform_bytes = vec![0_u8; resources.uniform_buffer_bytes as usize];
        let mut draws = Vec::with_capacity(plan.slices.len());
        let mut vertex_start = 0_usize;
        let mut index_start = 0_usize;
        for (slot, slice) in plan.slices.iter().enumerate() {
            let uniform_offset = resources.uniform_stride * slot as u64;
            write_uniform_bytes(
                &mut uniform_bytes,
                uniform_offset,
                &StageGpuUniforms::for_slice(slice),
            );
            draws.push(StageDraw {
                index_range: index_start as u32..(index_start + slice.indices.len()) as u32,
                base_vertex: vertex_start as i32,
                uniform_offset: uniform_offset as u32,
            });
            vertex_start += slice.vertices.len();
            index_start += slice.indices.len();
        }
        let (effective_route, output_identification) = endpoint_runtime_state(&plan, tools);
        let surface_uniform_offset = resources.uniform_stride * plan.slices.len() as u64;
        write_uniform_bytes(
            &mut uniform_bytes,
            surface_uniform_offset,
            &StageGpuUniforms::for_surface(effective_route, output_identification, &plan.id),
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("StageMap endpoint uniforms"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StageMap endpoint uniform bind group"),
            layout: &pipelines.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: NonZeroU64::new(STAGE_UNIFORM_BYTES),
                }),
            }],
        });
        let present_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StageMap endpoint physical surface source"),
            layout: &pipelines.source_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                },
            ],
        });
        Self {
            id: plan.id,
            binding,
            output_size: plan.output_size,
            refresh_millihz: plan.refresh_millihz,
            authored_route: plan.route,
            effective_route,
            output_identification,
            texture,
            view,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniform_bind_group,
            present_bind_group,
            draws: draws.into_boxed_slice(),
            surface_uniform_offset: surface_uniform_offset as u32,
            uniform_stride: resources.uniform_stride,
            frame_status: StageEndpointFrameStatus::NotRendered,
            resources,
        }
    }

    fn update_tools(&mut self, queue: &wgpu::Queue, tools: &StageToolState) {
        let plan = StageEndpointPlan {
            id: self.id.clone(),
            output_size: self.output_size,
            refresh_millihz: self.refresh_millihz,
            route: self.authored_route.clone(),
            slices: Vec::new(),
        };
        let (route, output_identification) = endpoint_runtime_state(&plan, tools);
        if route == self.effective_route && output_identification == self.output_identification {
            return;
        }
        self.effective_route = route;
        self.output_identification = output_identification;
        let uniforms = StageGpuUniforms::for_surface(route, output_identification, &self.id);
        queue.write_buffer(
            &self.uniform_buffer,
            u64::from(self.surface_uniform_offset),
            bytemuck::bytes_of(&uniforms),
        );
    }
}

fn write_uniform_bytes(bytes: &mut [u8], offset: u64, uniforms: &StageGpuUniforms) {
    let start = offset as usize;
    let end = start + std::mem::size_of::<StageGpuUniforms>();
    bytes[start..end].copy_from_slice(bytemuck::bytes_of(uniforms));
}

fn endpoint_runtime_state(
    endpoint: &StageEndpointPlan,
    tools: &StageToolState,
) -> (EffectiveRoute, bool) {
    let decision = tools.decision_for(&StageSurface::PhysicalOutput(endpoint.id.clone()));
    let route = if decision.substitute_with_test_card {
        EffectiveRoute::TestCard(tools.test_card())
    } else {
        match endpoint.route {
            StageRoute::Program => EffectiveRoute::Program,
            StageRoute::Blackout => EffectiveRoute::Blackout,
            StageRoute::TestCard { mode } => EffectiveRoute::TestCard(mode),
        }
    };
    (route, decision.overlay_output_identification)
}

pub(crate) struct StageEndpointOutput<'a> {
    pub endpoint_id: &'a OutputEndpointId,
    pub binding: &'a OutputBinding,
    pub output_size: [u32; 2],
    pub refresh_millihz: u32,
    pub format: wgpu::TextureFormat,
    pub texture: &'a wgpu::Texture,
    pub view: &'a wgpu::TextureView,
    pub frame_status: StageEndpointFrameStatus,
}

/// One successfully acquired host-owned physical surface view. The host keeps
/// ownership of its `SurfaceTexture` and presents it only after queue submit.
pub(crate) struct StageEndpointSurfaceTarget<'a> {
    pub endpoint_id: &'a OutputEndpointId,
    pub view: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub dimensions: [u32; 2],
}

pub(crate) struct StageMapPresenter {
    pipelines: Option<StagePipelines>,
    sources: Box<[PreparedProgramSource]>,
    endpoints: Box<[PreparedStageEndpoint]>,
    preparation: Box<[StageEndpointPreparation]>,
    allocations: StagePresenterAllocationSnapshot,
    current_tools: StageToolState,
}

impl StageMapPresenter {
    pub(crate) fn prepare(
        device: &wgpu::Device,
        stage_map: &StageMap,
        tools: &StageToolState,
        program_sources: &[StageProgramSource<'_>],
        surface_formats: &[wgpu::TextureFormat],
        limits: StagePresenterLimits,
        endpoint_available: impl FnMut(&crate::stage_map::StageEndpoint) -> Result<(), String>,
    ) -> Result<Self, StagePresenterPrepareError> {
        for (index, source) in program_sources.iter().enumerate() {
            if program_sources[..index]
                .iter()
                .any(|candidate| candidate.id == source.id)
            {
                return Err(StagePresenterPrepareError::DuplicateProgramSource(
                    source.id,
                ));
            }
        }
        let mut unique_surface_formats = Vec::with_capacity(surface_formats.len());
        for format in surface_formats.iter().copied() {
            if !unique_surface_formats.contains(&format) {
                unique_surface_formats.push(format);
            }
        }
        if unique_surface_formats.len() > STAGE_PRESENTER_MAX_SURFACE_FORMATS {
            return Err(StagePresenterPrepareError::TooManySurfaceFormats(
                unique_surface_formats.len(),
            ));
        }
        let plan = StagePresenterPlan::build(
            stage_map,
            limits,
            program_sources.len(),
            endpoint_available,
        )?;
        let mut preparation = plan.preparation().collect::<Vec<_>>();
        if plan.resources.ready_endpoints == 0 {
            return Ok(Self {
                pipelines: None,
                sources: Box::new([]),
                endpoints: Box::new([]),
                preparation: preparation.into_boxed_slice(),
                allocations: StagePresenterAllocationSnapshot::default(),
                current_tools: tools.clone(),
            });
        }

        let pipelines = scoped_device_build(device, || {
            StagePipelines::build(device, &unique_surface_formats)
        })
        .map_err(StagePresenterPrepareError::ResourceCreation)?;
        let sources = scoped_device_build(device, || {
            program_sources
                .iter()
                .map(|source| PreparedProgramSource {
                    id: source.id,
                    bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("StageMap prepared Program source"),
                        layout: &pipelines.source_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(source.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                            },
                        ],
                    }),
                })
                .collect::<Vec<_>>()
        })
        .map_err(StagePresenterPrepareError::ResourceCreation)?;

        let mut endpoints = Vec::with_capacity(plan.resources.ready_endpoints);
        for planned in plan.endpoints {
            let PlannedEndpoint {
                id,
                binding,
                admission,
            } = planned;
            let PlannedEndpointAdmission::Ready {
                plan: endpoint_plan,
                resources,
            } = admission
            else {
                continue;
            };
            match scoped_device_build(device, || {
                PreparedStageEndpoint::build(
                    device,
                    &pipelines,
                    endpoint_plan,
                    binding,
                    resources,
                    tools,
                )
            }) {
                Ok(endpoint) => endpoints.push(endpoint),
                Err(error) => {
                    if let Some(status) = preparation
                        .iter_mut()
                        .find(|status| status.endpoint_id == id)
                    {
                        status.status = StageEndpointPreparationStatus::Rejected(
                            StageEndpointPrepareError::ResourceCreation(error),
                        );
                    }
                }
            }
        }
        let allocations = allocation_snapshot(&pipelines, &sources, &endpoints);
        Ok(Self {
            pipelines: Some(pipelines),
            sources: sources.into_boxed_slice(),
            endpoints: endpoints.into_boxed_slice(),
            preparation: preparation.into_boxed_slice(),
            allocations,
            current_tools: tools.clone(),
        })
    }

    pub(crate) fn update_tools(&mut self, queue: &wgpu::Queue, tools: &StageToolState) {
        if &self.current_tools == tools {
            return;
        }
        for endpoint in &mut self.endpoints {
            endpoint.update_tools(queue, tools);
        }
        self.current_tools = tools.clone();
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        program_source: Option<StageProgramSourceId>,
    ) -> StagePresenterFrameMetrics {
        let mut metrics = StagePresenterFrameMetrics::default();
        let Some(pipelines) = &self.pipelines else {
            return metrics;
        };
        let source = program_source
            .and_then(|source| self.sources.iter().find(|candidate| candidate.id == source));
        for endpoint in &mut self.endpoints {
            endpoint.frame_status = StageEndpointFrameStatus::Presented;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("StageMap isolated endpoint presentation"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &endpoint.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            match endpoint.effective_route {
                EffectiveRoute::Program => {
                    metrics.program_endpoints += 1;
                    let Some(source) = source else {
                        endpoint.frame_status = StageEndpointFrameStatus::ProgramSourceUnavailable;
                        metrics.program_source_unavailable += 1;
                        if endpoint.output_identification {
                            encode_surface_overlay(&mut pass, pipelines, endpoint);
                        }
                        continue;
                    };
                    if let (Some(vertices), Some(indices)) =
                        (&endpoint.vertex_buffer, &endpoint.index_buffer)
                    {
                        pass.set_pipeline(&pipelines.slice);
                        pass.set_bind_group(0, &source.bind_group, &[]);
                        pass.set_vertex_buffer(0, vertices.slice(..));
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                        for draw in &endpoint.draws {
                            pass.set_bind_group(
                                1,
                                &endpoint.uniform_bind_group,
                                &[draw.uniform_offset],
                            );
                            pass.draw_indexed(draw.index_range.clone(), draw.base_vertex, 0..1);
                        }
                    }
                    if endpoint.output_identification {
                        encode_surface_overlay(&mut pass, pipelines, endpoint);
                    }
                }
                EffectiveRoute::Blackout => {
                    metrics.blackout_endpoints += 1;
                    if endpoint.output_identification {
                        encode_surface_overlay(&mut pass, pipelines, endpoint);
                    }
                }
                EffectiveRoute::TestCard(_) => {
                    metrics.test_card_endpoints += 1;
                    pass.set_pipeline(&pipelines.surface_replace);
                    pass.set_bind_group(0, &pipelines.empty_bind_group, &[]);
                    pass.set_bind_group(
                        1,
                        &endpoint.uniform_bind_group,
                        &[endpoint.surface_uniform_offset],
                    );
                    pass.draw(0..3, 0..1);
                }
            }
            metrics.presented_endpoints += 1;
        }
        metrics
    }

    /// Blit named endpoint textures into independently acquired physical
    /// surfaces. An absent or invalid target is skipped without touching any
    /// other endpoint or any creative/audience surface.
    pub(crate) fn encode_surface_targets(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &[StageEndpointSurfaceTarget<'_>],
    ) -> StageSurfaceFrameMetrics {
        let mut metrics = StageSurfaceFrameMetrics::default();
        let Some(pipelines) = &self.pipelines else {
            metrics.missing_endpoints = targets.len().min(u32::MAX as usize) as u32;
            return metrics;
        };
        for (target_index, target) in targets.iter().take(MAX_OUTPUT_ENDPOINTS).enumerate() {
            if targets[..target_index]
                .iter()
                .any(|prior| prior.endpoint_id == target.endpoint_id)
            {
                metrics.duplicate_targets += 1;
                continue;
            }
            let Some(endpoint) = self
                .endpoints
                .iter()
                .find(|endpoint| &endpoint.id == target.endpoint_id)
            else {
                metrics.missing_endpoints += 1;
                continue;
            };
            if !matches!(&endpoint.binding, OutputBinding::Monitor { .. }) {
                metrics.unassigned_endpoints += 1;
                continue;
            }
            if endpoint.output_size != target.dimensions {
                metrics.dimension_mismatches += 1;
                continue;
            }
            let Some(surface_pipeline) = pipelines
                .surface_present
                .iter()
                .find(|pipeline| pipeline.format == target.format)
            else {
                metrics.unsupported_formats += 1;
                continue;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("StageMap named physical surface presentation"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&surface_pipeline.pipeline);
            pass.set_bind_group(0, &endpoint.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
            metrics.presented_surfaces += 1;
        }
        metrics.excess_targets = targets.len().saturating_sub(MAX_OUTPUT_ENDPOINTS) as u32;
        metrics
    }

    pub(crate) fn preparation(&self) -> &[StageEndpointPreparation] {
        &self.preparation
    }

    pub(crate) const fn allocation_snapshot(&self) -> StagePresenterAllocationSnapshot {
        self.allocations
    }

    pub(crate) fn outputs(&self) -> impl ExactSizeIterator<Item = StageEndpointOutput<'_>> + '_ {
        self.endpoints.iter().map(|endpoint| StageEndpointOutput {
            endpoint_id: &endpoint.id,
            binding: &endpoint.binding,
            output_size: endpoint.output_size,
            refresh_millihz: endpoint.refresh_millihz,
            format: STAGE_ENDPOINT_FORMAT,
            texture: &endpoint.texture,
            view: &endpoint.view,
            frame_status: endpoint.frame_status,
        })
    }

    pub(crate) fn output(&self, id: &OutputEndpointId) -> Option<StageEndpointOutput<'_>> {
        self.outputs().find(|output| output.endpoint_id == id)
    }
}

fn encode_surface_overlay<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    pipelines: &'pass StagePipelines,
    endpoint: &'pass PreparedStageEndpoint,
) {
    pass.set_pipeline(&pipelines.surface_overlay);
    pass.set_bind_group(0, &pipelines.empty_bind_group, &[]);
    pass.set_bind_group(
        1,
        &endpoint.uniform_bind_group,
        &[endpoint.surface_uniform_offset],
    );
    pass.draw(0..3, 0..1);
}

fn allocation_snapshot(
    pipelines: &StagePipelines,
    sources: &[PreparedProgramSource],
    endpoints: &[PreparedStageEndpoint],
) -> StagePresenterAllocationSnapshot {
    StagePresenterAllocationSnapshot {
        textures: endpoints.len() as u64,
        views: endpoints.len() as u64,
        buffers: endpoints
            .iter()
            .map(|endpoint| endpoint.resources.buffers)
            .sum(),
        bind_groups: (sources.len() + endpoints.len() * 2 + 1) as u64,
        bind_group_layouts: 3,
        pipeline_layouts: 3,
        pipelines: 3 + pipelines.surface_present.len() as u64,
        shader_modules: 1,
        samplers: 1,
        texture_bytes: endpoints
            .iter()
            .map(|endpoint| endpoint.resources.output_texture_bytes)
            .sum(),
        buffer_bytes: endpoints
            .iter()
            .map(|endpoint| {
                endpoint.resources.mesh_buffer_bytes + endpoint.resources.uniform_buffer_bytes
            })
            .sum(),
    }
}

fn scoped_device_build<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> Result<T, String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let built = build();
    let error = [
        ("out of memory", pollster::block_on(out_of_memory.pop())),
        ("internal/backend", pollster::block_on(internal.pop())),
        ("validation", pollster::block_on(validation.pop())),
    ]
    .into_iter()
    .find_map(|(kind, error)| error.map(|error| format!("{kind}: {error}")));
    match error {
        Some(error) => Err(bounded_diagnostic(error)),
        None => Ok(built),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_map::{
        NormalizedQuad, NormalizedRect, OutputBinding, StageEndpoint, StageGeometry,
    };

    fn endpoint(id: &str, size: [u32; 2]) -> StageEndpoint {
        StageEndpoint {
            id: OutputEndpointId::parse(id).unwrap(),
            name: format!("Output {id}"),
            enabled: true,
            binding: OutputBinding::Unassigned,
            output_size: size,
            refresh_millihz: 60_000,
            route: StageRoute::Program,
            slices: Vec::new(),
        }
    }

    fn map_with_identity_slices(endpoints: Vec<StageEndpoint>) -> StageMap {
        let mut stage_map = StageMap::default();
        for endpoint in endpoints {
            let id = endpoint.id.clone();
            stage_map.add_endpoint(endpoint).unwrap();
            stage_map
                .add_slice(&id, format!("Slice {id}"), StageGeometry::default())
                .unwrap();
        }
        stage_map
    }

    fn test_limits(max_bytes: u64) -> StagePresenterLimits {
        StagePresenterLimits::bounded(
            StageDeviceLimits {
                max_dimension: 8_192,
                max_pixels_per_endpoint: u64::MAX,
                max_vertices_per_endpoint: 512,
            },
            max_bytes,
            1 << 20,
            256,
            2,
        )
    }

    #[test]
    fn cpu_plan_uses_frozen_meshes_and_accounts_every_persistent_byte() {
        let mut first = endpoint("first", [1920, 1080]);
        first.slices.clear();
        let mut second = endpoint("second", [1280, 720]);
        second.slices.clear();
        let stage_map = map_with_identity_slices(vec![first, second]);
        let plan = StagePresenterPlan::build(
            &stage_map,
            test_limits(STAGE_PRESENTER_HARD_MAX_GPU_BYTES),
            2,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(plan.resources.ready_endpoints, 2);
        assert_eq!(plan.resources.output_pixels, 1920 * 1080 + 1280 * 720);
        assert_eq!(
            plan.resources.output_texture_bytes,
            plan.resources.output_pixels * 4
        );
        assert_eq!(plan.resources.mesh_buffer_bytes, 2 * (4 * 16 + 6 * 2));
        assert_eq!(plan.resources.uniform_buffer_bytes, 2 * 2 * 256);
        assert_eq!(plan.resources.buffers, 6);
        assert_eq!(plan.resources.bind_groups, 7);
        assert_eq!(plan.preparation().count(), 2);
    }

    #[test]
    fn total_budget_and_unavailable_endpoints_reject_in_isolation() {
        let mut admitted = endpoint("admitted", [4, 4]);
        admitted.slices.clear();
        let mut too_large_for_remaining = endpoint("too-large", [8, 8]);
        too_large_for_remaining.slices.clear();
        let mut later_small = endpoint("later-small", [1, 1]);
        later_small.slices.clear();
        let missing = endpoint("missing", [1, 1]);
        let estimate_map = map_with_identity_slices(vec![admitted.clone()]);
        let endpoint_bytes = estimate_endpoint(
            &estimate_map.evaluate_isolated(test_limits(u64::MAX).stage, |_| Ok(()))[0]
                .result
                .clone()
                .unwrap()
                .unwrap(),
            test_limits(u64::MAX),
        )
        .unwrap()
        .total_gpu_bytes;
        let mut stage_map =
            map_with_identity_slices(vec![admitted, too_large_for_remaining, later_small]);
        stage_map.add_endpoint(missing).unwrap();
        let plan = StagePresenterPlan::build(
            &stage_map,
            test_limits(endpoint_bytes + 700),
            1,
            |endpoint| {
                if endpoint.id.as_str() == "missing" {
                    Err(format!("disconnected\n{}", "x".repeat(2_000)))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap();
        let statuses = plan.preparation().collect::<Vec<_>>();
        assert!(matches!(
            statuses[0].status,
            StageEndpointPreparationStatus::Ready { .. }
        ));
        assert!(matches!(
            statuses[1].status,
            StageEndpointPreparationStatus::Rejected(
                StageEndpointPrepareError::TotalGpuBudgetExceeded { .. }
            )
        ));
        assert!(matches!(
            statuses[2].status,
            StageEndpointPreparationStatus::Ready { .. }
        ));
        let StageEndpointPreparationStatus::Rejected(StageEndpointPrepareError::Domain(
            StageEndpointRuntimeError::Unavailable(message),
        )) = &statuses[3].status
        else {
            panic!("missing endpoint must retain a bounded unavailable diagnostic")
        };
        assert!(message.len() <= STAGE_PRESENTER_DIAGNOSTIC_MAX_BYTES);
        assert!(!message.chars().any(char::is_control));
        assert_eq!(plan.resources.ready_endpoints, 2);
    }

    #[test]
    fn tool_decisions_are_exact_endpoint_overrides_and_never_change_authored_routes() {
        let selected = OutputEndpointId::parse("selected").unwrap();
        let other = OutputEndpointId::parse("other").unwrap();
        let mut tools = StageToolState::default();
        tools
            .set_test_card(TestCardMode::Grid, Some(selected.clone()))
            .unwrap();
        tools
            .set_output_identification(true, Some(selected.clone()))
            .unwrap();
        let selected_plan = StageEndpointPlan {
            id: selected,
            output_size: [1, 1],
            refresh_millihz: 60_000,
            route: StageRoute::Program,
            slices: Vec::new(),
        };
        let other_plan = StageEndpointPlan {
            id: other,
            route: StageRoute::Blackout,
            ..selected_plan.clone()
        };
        assert_eq!(
            endpoint_runtime_state(&selected_plan, &tools),
            (EffectiveRoute::TestCard(TestCardMode::Grid), true)
        );
        assert_eq!(
            endpoint_runtime_state(&other_plan, &tools),
            (EffectiveRoute::Blackout, false)
        );
        assert_eq!(selected_plan.route, StageRoute::Program);
    }

    #[test]
    fn hard_caps_and_source_identity_are_closed_before_gpu_allocation() {
        let limits = StagePresenterLimits::bounded(
            StageDeviceLimits::default(),
            u64::MAX,
            u64::MAX,
            256,
            usize::MAX,
        );
        assert_eq!(
            limits.max_total_gpu_bytes,
            STAGE_PRESENTER_HARD_MAX_GPU_BYTES
        );
        assert_eq!(
            limits.max_program_sources,
            STAGE_PRESENTER_MAX_PROGRAM_SOURCES
        );
        let plan = StagePresenterPlan::build(
            &StageMap::default(),
            limits,
            STAGE_PRESENTER_MAX_PROGRAM_SOURCES + 1,
            |_| Ok(()),
        );
        assert!(matches!(
            plan,
            Err(StagePresenterPrepareError::TooManyProgramSources(_))
        ));
    }

    #[test]
    fn gpu_uniform_pack_preserves_homography_mask_and_linear_calibration() {
        let stage_map = map_with_identity_slices(vec![endpoint("uniform", [10, 10])]);
        let mut slice = stage_map.evaluate_isolated(StageDeviceLimits::default(), |_| Ok(()))[0]
            .result
            .clone()
            .unwrap()
            .unwrap()
            .slices
            .remove(0);
        for (vertex, output_uv) in
            slice
                .vertices
                .iter_mut()
                .zip([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]])
        {
            vertex.output_uv = output_uv;
        }
        slice.output_to_source = Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        slice.mask = StageMask::Polygon {
            points: vec![[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            invert: true,
            softness: 0.05,
        };
        slice.calibration = StageCalibration {
            opacity: 0.5,
            brightness: 1.2,
            contrast: 0.8,
            gamma: 2.0,
            gain: [1.0, 0.9, 0.8],
            black_level: [0.01, 0.02, 0.03],
        };
        let uniforms = StageGpuUniforms::for_slice(&slice);
        assert_eq!(uniforms.homography_0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(uniforms.homography_2, [7.0, 8.0, 9.0, 0.0]);
        assert_eq!(uniforms.modes[..3], [1, 2, 4]);
        assert_eq!(uniforms.black_invert, [0.01, 0.02, 0.03, 1.0]);
        assert_eq!(uniforms.bounds, [0.1, 0.2, 0.9, 0.8]);
    }

    fn gpu_device(label: &'static str) -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("StageMap GPU adapter");
        let info = adapter.get_info();
        eprintln!(
            "StageMap physical GPU receipt: name={:?}, backend={:?}, device_type={:?}, driver={:?}, driver_info={:?}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("StageMap GPU device")
    }

    fn rgba_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        pixels: &[u8],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        assert_eq!(pixels.len(), size[0] as usize * size[1] as usize * 4);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("StageMap GPU fixture Program"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: STAGE_ENDPOINT_FORMAT,
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
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size[0] * 4),
                rows_per_image: Some(size[1]),
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn target_texture_with_format(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("StageMap physical surface fixture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn target_texture(device: &wgpu::Device, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
        target_texture_with_format(device, size, STAGE_ENDPOINT_FORMAT)
    }

    fn read_rgba(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: [u32; 2],
    ) -> Vec<u8> {
        let padded_row = (size[0] * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StageMap GPU fixture readback"),
            size: u64::from(padded_row) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("StageMap GPU fixture readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("StageMap GPU fixture wait");
        receive.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity(size[0] as usize * size[1] as usize * 4);
        for row in mapped
            .chunks_exact(padded_row as usize)
            .take(size[1] as usize)
        {
            rgba.extend_from_slice(&row[..size[0] as usize * 4]);
        }
        drop(mapped);
        staging.unmap();
        rgba
    }

    fn pixel_at(bytes: &[u8], size: [u32; 2], x: usize, y: usize) -> &[u8] {
        &bytes[(y * size[0] as usize + x) * 4..(y * size[0] as usize + x + 1) * 4]
    }

    fn gpu_map(id: &str, size: [u32; 2], geometry: StageGeometry, mask: StageMask) -> StageMap {
        let mut endpoint = endpoint(id, size);
        endpoint.binding = OutputBinding::Monitor {
            selector: format!("monitor-{id}"),
        };
        let endpoint_id = endpoint.id.clone();
        let mut stage_map = StageMap::default();
        stage_map.add_endpoint(endpoint).unwrap();
        stage_map
            .add_slice(&endpoint_id, "GPU slice", geometry)
            .unwrap();
        stage_map.endpoints[0].slices[0].mask = mask;
        stage_map
    }

    fn prepare_gpu_presenter(
        device: &wgpu::Device,
        stage_map: &StageMap,
        source_view: &wgpu::TextureView,
        endpoint_available: impl FnMut(&StageEndpoint) -> Result<(), String>,
    ) -> StageMapPresenter {
        StageMapPresenter::prepare(
            device,
            stage_map,
            &StageToolState::default(),
            &[StageProgramSource {
                id: StageProgramSourceId::new(0),
                view: source_view,
            }],
            &[STAGE_ENDPOINT_FORMAT, wgpu::TextureFormat::Bgra8UnormSrgb],
            StagePresenterLimits::for_device(device),
            endpoint_available,
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn physical_gpu_identity_slice_is_exact_and_warm() {
        let (device, queue) = gpu_device("StageMap identity fixture");
        let size = [4, 4];
        let pixels = (0..16_u8)
            .flat_map(|pixel| [pixel * 11, 255 - pixel * 7, pixel * 3, 255])
            .collect::<Vec<_>>();
        let (_source, source_view) = rgba_texture(&device, &queue, size, &pixels);
        let stage_map = gpu_map("identity", size, StageGeometry::default(), StageMask::None);
        let mut presenter = prepare_gpu_presenter(&device, &stage_map, &source_view, |_| Ok(()));
        let warmed = presenter.allocation_snapshot();
        for _ in 0..2 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("StageMap identity fixture encoder"),
            });
            let metrics = presenter.encode(&mut encoder, Some(StageProgramSourceId::new(0)));
            assert_eq!(metrics.presented_endpoints, 1);
            queue.submit(Some(encoder.finish()));
            assert_eq!(presenter.allocation_snapshot(), warmed);
        }
        let output = presenter.output(&stage_map.endpoints[0].id).unwrap();
        let actual = read_rgba(&device, &queue, output.texture, size);
        for (actual, expected) in actual.iter().zip(&pixels) {
            assert!(
                actual.abs_diff(*expected) <= 1,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn physical_gpu_perspective_quad_uses_projective_mapping() {
        let (device, queue) = gpu_device("StageMap perspective fixture");
        let size = [8, 8];
        let pixels = (0..size[1])
            .flat_map(|y| (0..size[0]).flat_map(move |x| [(x * 31) as u8, (y * 31) as u8, 40, 255]))
            .collect::<Vec<_>>();
        let (_source, source_view) = rgba_texture(&device, &queue, size, &pixels);
        let geometry = StageGeometry::PerspectiveQuad {
            source: NormalizedRect::default(),
            output: NormalizedQuad {
                points: [[0.25, 0.0], [0.75, 0.0], [1.0, 1.0], [0.0, 1.0]],
            },
        };
        let stage_map = gpu_map("perspective", size, geometry, StageMask::None);
        let mut presenter = prepare_gpu_presenter(&device, &stage_map, &source_view, |_| Ok(()));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("StageMap perspective fixture encoder"),
        });
        presenter.encode(&mut encoder, Some(StageProgramSourceId::new(0)));
        queue.submit(Some(encoder.finish()));
        let output = presenter.output(&stage_map.endpoints[0].id).unwrap();
        let actual = read_rgba(&device, &queue, output.texture, size);
        let pixel = |x: usize, y: usize| &actual[(y * 8 + x) * 4..(y * 8 + x + 1) * 4];
        assert_eq!(pixel(0, 0), &[0, 0, 0, 255]);
        assert!(pixel(4, 4)[0] > 80 && pixel(4, 4)[1] > 80);
        assert!(pixel(0, 7)[1] > 180);
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn physical_gpu_polygon_mask_is_bounded_and_softness_safe() {
        let (device, queue) = gpu_device("StageMap mask fixture");
        let size = [8, 8];
        let pixels = vec![255_u8; size[0] as usize * size[1] as usize * 4];
        let (_source, source_view) = rgba_texture(&device, &queue, size, &pixels);
        let stage_map = gpu_map(
            "mask",
            size,
            StageGeometry::default(),
            StageMask::Polygon {
                points: vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
                invert: false,
                softness: 0.05,
            },
        );
        let mut presenter = prepare_gpu_presenter(&device, &stage_map, &source_view, |_| Ok(()));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("StageMap mask fixture encoder"),
        });
        presenter.encode(&mut encoder, Some(StageProgramSourceId::new(0)));
        queue.submit(Some(encoder.finish()));
        let output = presenter.output(&stage_map.endpoints[0].id).unwrap();
        let actual = read_rgba(&device, &queue, output.texture, size);
        let pixel = |x: usize, y: usize| &actual[(y * 8 + x) * 4..(y * 8 + x + 1) * 4];
        assert_eq!(pixel(0, 0), &[0, 0, 0, 255]);
        assert!(pixel(4, 4)[0] > 250);
        assert_eq!(pixel(7, 7), &[0, 0, 0, 255]);
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn physical_gpu_multi_output_and_surface_failures_are_isolated() {
        let (device, queue) = gpu_device("StageMap multi-output fixture");
        let size = [2, 2];
        let pixels = [255, 0, 0, 255].repeat(4);
        let (source, source_view) = rgba_texture(&device, &queue, size, &pixels);
        let mut stage_map = gpu_map("program", size, StageGeometry::default(), StageMask::None);
        let mut test_card = endpoint("test-card", size);
        test_card.binding = OutputBinding::Monitor {
            selector: "monitor-test-card".into(),
        };
        test_card.route = StageRoute::TestCard {
            mode: TestCardMode::SmpteBars,
        };
        stage_map.add_endpoint(test_card).unwrap();
        let mut missing = endpoint("missing", size);
        missing.binding = OutputBinding::Monitor {
            selector: "monitor-missing".into(),
        };
        stage_map.add_endpoint(missing).unwrap();

        let mut presenter = prepare_gpu_presenter(&device, &stage_map, &source_view, |endpoint| {
            if endpoint.id.as_str() == "missing" {
                Err("surface acquisition failed".into())
            } else {
                Ok(())
            }
        });
        assert_eq!(presenter.outputs().len(), 2);
        assert!(matches!(
            presenter.preparation()[2].status,
            StageEndpointPreparationStatus::Rejected(_)
        ));
        let (program_surface, program_surface_view) = target_texture(&device, size);
        let (test_card_surface, test_card_surface_view) =
            target_texture_with_format(&device, size, wgpu::TextureFormat::Bgra8UnormSrgb);
        let warmed = presenter.allocation_snapshot();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("StageMap multi-output fixture encoder"),
        });
        let endpoint_metrics = presenter.encode(&mut encoder, Some(StageProgramSourceId::new(0)));
        assert_eq!(endpoint_metrics.presented_endpoints, 2);
        let surface_metrics = presenter.encode_surface_targets(
            &mut encoder,
            &[
                StageEndpointSurfaceTarget {
                    endpoint_id: &stage_map.endpoints[0].id,
                    view: &program_surface_view,
                    format: STAGE_ENDPOINT_FORMAT,
                    dimensions: size,
                },
                StageEndpointSurfaceTarget {
                    endpoint_id: &stage_map.endpoints[1].id,
                    view: &test_card_surface_view,
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    dimensions: size,
                },
                StageEndpointSurfaceTarget {
                    endpoint_id: &stage_map.endpoints[2].id,
                    view: &test_card_surface_view,
                    format: STAGE_ENDPOINT_FORMAT,
                    dimensions: size,
                },
            ],
        );
        assert_eq!(surface_metrics.presented_surfaces, 2);
        assert_eq!(surface_metrics.missing_endpoints, 1);
        queue.submit(Some(encoder.finish()));
        assert_eq!(presenter.allocation_snapshot(), warmed);

        assert_eq!(read_rgba(&device, &queue, &program_surface, size), pixels);
        let test_card_output = presenter.output(&stage_map.endpoints[1].id).unwrap();
        let test_card_rgba = read_rgba(&device, &queue, test_card_output.texture, size);
        let test_card_bgra = read_rgba(&device, &queue, &test_card_surface, size);
        for (rgba, bgra) in test_card_rgba
            .chunks_exact(4)
            .zip(test_card_bgra.chunks_exact(4))
        {
            assert!(rgba[0].abs_diff(bgra[2]) <= 1);
            assert!(rgba[1].abs_diff(bgra[1]) <= 1);
            assert!(rgba[2].abs_diff(bgra[0]) <= 1);
            assert_eq!(rgba[3], bgra[3]);
        }
        assert_eq!(read_rgba(&device, &queue, &source, size), pixels);
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn physical_gpu_edge_calibration_test_card_and_output_id_are_visible() {
        let (device, queue) = gpu_device("StageMap calibration tools fixture");
        let size = [64, 64];
        let pixels = [255_u8, 255, 255, 255].repeat(size[0] as usize * size[1] as usize);
        let (_source, source_view) = rgba_texture(&device, &queue, size, &pixels);
        let mut stage_map = gpu_map(
            "calibrated",
            size,
            StageGeometry::default(),
            StageMask::EdgeFeather { softness: [0.1; 4] },
        );
        stage_map.endpoints[0].slices[0].calibration = StageCalibration {
            brightness: 0.5,
            gain: [1.0, 0.5, 0.25],
            ..StageCalibration::default()
        };
        let mut tools_endpoint = endpoint("tools", size);
        tools_endpoint.binding = OutputBinding::Monitor {
            selector: "monitor-tools".into(),
        };
        let tools_id = tools_endpoint.id.clone();
        stage_map.add_endpoint(tools_endpoint).unwrap();
        let mut tools = StageToolState::default();
        tools
            .set_test_card(TestCardMode::Grid, Some(tools_id.clone()))
            .unwrap();
        tools
            .set_output_identification(true, Some(tools_id.clone()))
            .unwrap();
        let mut presenter = StageMapPresenter::prepare(
            &device,
            &stage_map,
            &tools,
            &[StageProgramSource {
                id: StageProgramSourceId::new(0),
                view: &source_view,
            }],
            &[STAGE_ENDPOINT_FORMAT],
            StagePresenterLimits::for_device(&device),
            |_| Ok(()),
        )
        .unwrap();
        let warmed = presenter.allocation_snapshot();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("StageMap calibration tools fixture encoder"),
        });
        let metrics = presenter.encode(&mut encoder, Some(StageProgramSourceId::new(0)));
        assert_eq!(metrics.program_endpoints, 1);
        assert_eq!(metrics.test_card_endpoints, 1);
        assert_eq!(metrics.presented_endpoints, 2);
        queue.submit(Some(encoder.finish()));
        assert_eq!(presenter.allocation_snapshot(), warmed);

        let calibrated = presenter.output(&stage_map.endpoints[0].id).unwrap();
        let calibrated = read_rgba(&device, &queue, calibrated.texture, size);
        assert!(pixel_at(&calibrated, size, 0, 0)[..3]
            .iter()
            .all(|channel| *channel < 10));
        let center = pixel_at(&calibrated, size, 32, 32);
        for (actual, expected) in center.iter().zip([188_u8, 137, 99, 255]) {
            assert!(
                actual.abs_diff(expected) <= 2,
                "actual={actual}, expected={expected}"
            );
        }

        let tools_output = presenter.output(&tools_id).unwrap();
        let tools_bytes = read_rgba(&device, &queue, tools_output.texture, size);
        let hash = endpoint_hash(&tools_id);
        let encode_srgb = |linear: f32| {
            let encoded = if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 255.0).round() as u8
        };
        let expected_identifier = [
            encode_srgb(0.25 + 0.75 * (hash & 255) as f32 / 255.0),
            encode_srgb(0.25 + 0.75 * ((hash >> 8) & 255) as f32 / 255.0),
            encode_srgb(0.25 + 0.75 * ((hash >> 16) & 255) as f32 / 255.0),
            255,
        ];
        for (actual, expected) in pixel_at(&tools_bytes, size, 0, 0)
            .iter()
            .zip(expected_identifier)
        {
            assert!(
                actual.abs_diff(expected) <= 1,
                "actual={actual}, expected={expected}"
            );
        }
        assert!(tools_bytes
            .chunks_exact(4)
            .any(|candidate| candidate != pixel_at(&tools_bytes, size, 0, 0)));
    }
}
