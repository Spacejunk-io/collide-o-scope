//! Shared image-matte math and bounded scratch planning.
//!
//! This module deliberately contains no window or authored-layer state.  The
//! live renderer and offline exporter can therefore consume the same resolved
//! matte payload and the same admission limits.

#[cfg(test)]
use super::blend::composite_shader_source;
use super::blend::matte_composite_shader_source;

/// A deliberately small ceiling for full-frame materialized image taps.
///
/// `OneBelow` and `AllBelow` can normally consume an already available image;
/// this limit applies only to unique selected-layer pre/post-local taps that
/// must be rendered into scratch array layers.
pub const MAX_MATERIALIZED_IMAGE_TAPS: u32 = 64;

/// A host-independent ceiling for tap scratch.  Adapter limits are checked in
/// addition to this value so an unusually permissive driver cannot turn an
/// authored patch into an unbounded allocation request.
pub const MAX_IMAGE_TAP_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) const SELECTIVE_MATTE_TOPOLOGY_ERROR: &str =
    "selective per-layer VHS with image mattes is not a valid Milestone 1 topology";

/// Milestone 1 deliberately refuses the selective-VHS topology rather than
/// emitting unmasked slices or advancing program history with the wrong
/// image. Live holds its last accepted audience frame; export aborts the job
/// with the same visible status.
pub(crate) fn validate_selective_matte_topology(image_routing_active: bool) -> Result<(), String> {
    if image_routing_active {
        Err(SELECTIVE_MATTE_TOPOLOGY_ERROR.to_string())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatteResourceLimits {
    pub max_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_bytes: u64,
}

impl MatteResourceLimits {
    pub fn from_wgpu(limits: &wgpu::Limits) -> Self {
        Self {
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_texture_array_layers: limits.max_texture_array_layers,
            max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage,
            max_bytes: MAX_IMAGE_TAP_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatteResourcePlan {
    pub output_size: [u32; 2],
    pub tap_layers: u32,
    /// Exact RGBA8 payload for the materialized taps and one persistent
    /// previous-clean-program image. Driver padding is intentionally excluded.
    pub rgba_payload_bytes: u64,
}

impl MatteResourcePlan {
    pub fn validate(
        output_size: [u32; 2],
        materialized_taps: usize,
        limits: MatteResourceLimits,
    ) -> Result<Self, String> {
        let [width, height] = output_size;
        if width == 0 || height == 0 {
            return Err("image routing output dimensions cannot be zero".into());
        }
        if width > limits.max_texture_dimension_2d || height > limits.max_texture_dimension_2d {
            return Err(format!(
                "image routing output {width}x{height} exceeds the GPU 2D texture limit of {}",
                limits.max_texture_dimension_2d
            ));
        }
        let tap_layers = u32::try_from(materialized_taps)
            .map_err(|_| "image tap count does not fit the GPU array-layer domain".to_string())?;
        if tap_layers > MAX_MATERIALIZED_IMAGE_TAPS {
            return Err(format!(
                "image routing requests {tap_layers} materialized taps; the hard limit is {MAX_MATERIALIZED_IMAGE_TAPS}"
            ));
        }
        // The live allocation uses at least one array layer because a zero-
        // layer texture is invalid. No allocation is performed for a zero-tap
        // plan, so the adapter array check applies only when taps are present.
        if tap_layers > 0 && tap_layers > limits.max_texture_array_layers {
            return Err(format!(
                "image routing requests {tap_layers} tap layers; this GPU supports {}",
                limits.max_texture_array_layers
            ));
        }
        // The matte composite shader binds base, overlay, and donor.
        if limits.max_sampled_textures_per_shader_stage < 3 {
            return Err(format!(
                "image mattes require three sampled textures; this GPU supports {}",
                limits.max_sampled_textures_per_shader_stage
            ));
        }

        let frame_bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "image routing frame byte count overflowed".to_string())?;
        // Exactly one persistent program-history texture is part of a routed
        // compositor allocation, even before a ProgramHistory input is used.
        let texture_count = u64::from(tap_layers)
            .checked_add(1)
            .ok_or_else(|| "image routing texture count overflowed".to_string())?;
        let rgba_payload_bytes = frame_bytes
            .checked_mul(texture_count)
            .ok_or_else(|| "image routing scratch byte count overflowed".to_string())?;
        if rgba_payload_bytes > limits.max_bytes {
            return Err(format!(
                "image routing requires {rgba_payload_bytes} RGBA bytes; the bounded limit is {}",
                limits.max_bytes
            ));
        }

        Ok(Self {
            output_size,
            tap_layers,
            rgba_payload_bytes,
        })
    }
}

/// Shader-facing channel codes. Keep synchronized with `matte_composite.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MatteChannelCode {
    Alpha = 0,
    Luma = 1,
    Red = 2,
    Green = 3,
    Blue = 4,
}

/// Resolved, finite matte values consumed by both CPU references and the GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedMatteParams {
    pub channel: MatteChannelCode,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
    /// False means the authored donor was missing or rejected. It is distinct
    /// from a valid all-zero donor: invert is suppressed and the shaped field
    /// is zero, while amount remains a continuous dry/wet control.
    pub donor_valid: bool,
}

impl ResolvedMatteParams {
    pub fn sanitized(self) -> Self {
        Self {
            channel: self.channel,
            invert: self.invert,
            amount: finite_unit(self.amount),
            threshold: finite_unit(self.threshold),
            softness: finite_unit(self.softness),
            donor_valid: self.donor_valid,
        }
    }
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU reference for the matte shader. Inputs and output are straight-alpha.
#[cfg(test)]
pub fn apply_matte_alpha(
    overlay: [f32; 4],
    donor: [f32; 4],
    params: ResolvedMatteParams,
) -> [f32; 4] {
    let params = params.sanitized();
    let shaped = if params.donor_valid {
        let mut field = match params.channel {
            MatteChannelCode::Alpha => donor[3],
            MatteChannelCode::Luma => donor[0] * 0.2126 + donor[1] * 0.7152 + donor[2] * 0.0722,
            MatteChannelCode::Red => donor[0],
            MatteChannelCode::Green => donor[1],
            MatteChannelCode::Blue => donor[2],
        };
        field = finite_unit(field);
        if params.invert {
            field = 1.0 - field;
        }
        if params.softness <= f32::EPSILON {
            if field >= params.threshold {
                1.0
            } else {
                0.0
            }
        } else {
            let half_width = params.softness * 0.5;
            smoothstep(
                params.threshold - half_width,
                params.threshold + half_width,
                field,
            )
        }
    } else {
        // Missing is a defined zero field. Invert cannot resurrect it, and
        // amount=0 remains the exact dry/bypass endpoint.
        0.0
    };
    let admission = 1.0 + (shaped - 1.0) * params.amount;
    [
        overlay[0],
        overlay[1],
        overlay[2],
        finite_unit(overlay[3]) * admission,
    ]
}

/// Uniform block for the routed composite pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatteCompositeUniforms {
    pub opacity: f32,
    pub blend_mode: u32,
    pub channel: u32,
    pub invert: u32,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
    pub donor_valid: u32,
}

impl MatteCompositeUniforms {
    pub fn new(opacity: f32, blend_mode: u32, params: ResolvedMatteParams) -> Self {
        let params = params.sanitized();
        Self {
            opacity,
            blend_mode,
            channel: params.channel as u32,
            invert: u32::from(params.invert),
            amount: params.amount,
            threshold: params.threshold,
            softness: params.softness,
            donor_valid: u32::from(params.donor_valid),
        }
    }
}

const _: () = assert!(std::mem::size_of::<MatteCompositeUniforms>() == 32);

/// GPU pipeline kept separate from the legacy two-input compositor. Merely
/// constructing this pipeline does not allocate any full-frame tap resource.
pub(crate) struct MatteCompositePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub texture_layout: wgpu::BindGroupLayout,
    pub uniform_layout: wgpu::BindGroupLayout,
}

impl MatteCompositePipeline {
    pub fn build(device: &wgpu::Device, vertex_shader: &wgpu::ShaderModule) -> Self {
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matte Composite Texture BGL"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matte Composite Uniform BGL"),
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
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matte Composite Fragment"),
            source: wgpu::ShaderSource::Wgsl(matte_composite_shader_source()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matte Composite Pipeline Layout"),
            bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Matte Composite Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
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
        Self {
            pipeline,
            texture_layout,
            uniform_layout,
        }
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

/// Lazily-created output-sized selected-layer tap array.
pub(crate) struct ImageTapTexture {
    /// Kept alive for the slice views below.
    pub _texture: wgpu::Texture,
    pub views: Vec<wgpu::TextureView>,
}

impl ImageTapTexture {
    pub fn build(device: &wgpu::Device, output_size: [u32; 2], layers: u32) -> Option<Self> {
        if layers == 0 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Selected Image Tap Array"),
            size: wgpu::Extent3d {
                width: output_size[0],
                height: output_size[1],
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let views = (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Selected Image Tap Slice"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Some(Self {
            _texture: texture,
            views,
        })
    }
}

/// Exactly one persistent N-1 clean-program texture.
pub(crate) struct ProgramHistoryTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl ProgramHistoryTexture {
    pub fn build(device: &wgpu::Device, output_size: [u32; 2]) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Previous Clean Program"),
            size: wgpu::Extent3d {
                width: output_size[0],
                height: output_size[1],
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
        Self { texture, view }
    }
}

/// Shared lazy resource state used by live and offline compositors. Callers
/// own error scopes because live and export report GPU failures differently.
pub(crate) struct ImageRoutingGpuResources {
    pub output_size: [u32; 2],
    pub tap_layers: u32,
    pub taps: Option<ImageTapTexture>,
    pub history: ProgramHistoryTexture,
    pub history_valid: bool,
}

impl ImageRoutingGpuResources {
    pub fn build(device: &wgpu::Device, plan: MatteResourcePlan) -> Self {
        Self {
            output_size: plan.output_size,
            tap_layers: plan.tap_layers,
            taps: ImageTapTexture::build(device, plan.output_size, plan.tap_layers),
            history: ProgramHistoryTexture::build(device, plan.output_size),
            history_valid: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_matte_composite(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &MatteCompositePipeline,
    sampler: &wgpu::Sampler,
    base: &wgpu::TextureView,
    overlay: &wgpu::TextureView,
    donor: &wgpu::TextureView,
    output: &wgpu::TextureView,
    uniforms: MatteCompositeUniforms,
) {
    let bytes = bytemuck::bytes_of(&uniforms);
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matte Composite Uniforms"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buffer, 0, bytes);
    let texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matte Composite Textures BG"),
        layout: &pipeline.texture_layout,
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
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matte Composite Uniform BG"),
        layout: &pipeline.uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Routed Matte Composite Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
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
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, &texture_group, &[]);
    pass.set_bind_group(1, &uniform_group, &[]);
    pass.draw(0..3, 0..1);
}

pub(crate) fn encode_program_history_copy(
    encoder: &mut wgpu::CommandEncoder,
    clean_program: &wgpu::Texture,
    resources: &mut ImageRoutingGpuResources,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: clean_program,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &resources.history.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: resources.output_size[0],
            height: resources.output_size[1],
            depth_or_array_layers: 1,
        },
    );
    resources.history_valid = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::BlendMode;
    use crate::renderer::blend::composite_straight;

    const OVERLAY: [f32; 4] = [0.8, 0.4, 0.2, 0.8];

    fn params(channel: MatteChannelCode) -> ResolvedMatteParams {
        ResolvedMatteParams {
            channel,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 1.0,
            donor_valid: true,
        }
    }

    #[test]
    fn channel_reference_vectors_are_straight_alpha() {
        let donor = [0.25, 0.5, 0.75, 0.5];
        let alpha = apply_matte_alpha(OVERLAY, donor, params(MatteChannelCode::Alpha));
        let red = apply_matte_alpha(OVERLAY, donor, params(MatteChannelCode::Red));
        let green = apply_matte_alpha(OVERLAY, donor, params(MatteChannelCode::Green));
        let blue = apply_matte_alpha(OVERLAY, donor, params(MatteChannelCode::Blue));
        let luma = apply_matte_alpha(OVERLAY, donor, params(MatteChannelCode::Luma));

        // softness=1, threshold=.5 maps the unit field through smoothstep(0, 1).
        assert_eq!(&alpha[..3], &OVERLAY[..3]);
        assert!((alpha[3] - 0.4).abs() < 1.0e-6);
        assert!((red[3] - 0.125).abs() < 1.0e-6);
        assert!((green[3] - 0.4).abs() < 1.0e-6);
        assert!((blue[3] - 0.675).abs() < 1.0e-6);
        let luma_field = 0.25 * 0.2126 + 0.5 * 0.7152 + 0.75 * 0.0722;
        let expected_luma_alpha = 0.8 * smoothstep(0.0, 1.0, luma_field);
        assert!((luma[3] - expected_luma_alpha).abs() < 1.0e-6);
    }

    #[test]
    fn half_alpha_donor_halves_overlay_without_threshold_shaping() {
        let output = apply_matte_alpha(
            OVERLAY,
            [1.0, 1.0, 1.0, 0.5],
            ResolvedMatteParams {
                channel: MatteChannelCode::Alpha,
                invert: false,
                amount: 1.0,
                threshold: 0.0,
                softness: 0.0,
                donor_valid: true,
            },
        );
        // A zero-width threshold is deliberately binary. The continuous
        // half-alpha reference uses a full soft interval centered at 0.5.
        assert_eq!(output[3], OVERLAY[3]);
        let continuous = apply_matte_alpha(
            OVERLAY,
            [1.0, 1.0, 1.0, 0.5],
            ResolvedMatteParams {
                threshold: 0.5,
                softness: 1.0,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert!((continuous[3] - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn invert_amount_threshold_softness_and_missing_are_defined() {
        let inverted = apply_matte_alpha(
            OVERLAY,
            [0.25; 4],
            ResolvedMatteParams {
                invert: true,
                threshold: 0.5,
                softness: 0.0,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert_eq!(inverted[3], 0.8);

        let half_amount = apply_matte_alpha(
            OVERLAY,
            [0.0; 4],
            ResolvedMatteParams {
                amount: 0.5,
                threshold: 0.5,
                softness: 0.0,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert!((half_amount[3] - 0.4).abs() < 1.0e-6);

        let soft = apply_matte_alpha(
            OVERLAY,
            [0.5; 4],
            ResolvedMatteParams {
                threshold: 0.5,
                softness: 0.4,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert!((soft[3] - 0.4).abs() < 1.0e-6);

        let missing_full = apply_matte_alpha(
            OVERLAY,
            [1.0; 4],
            ResolvedMatteParams {
                invert: true,
                donor_valid: false,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert_eq!(missing_full, [0.8, 0.4, 0.2, 0.0]);
        let missing_bypass = apply_matte_alpha(
            OVERLAY,
            [1.0; 4],
            ResolvedMatteParams {
                invert: true,
                amount: 0.0,
                donor_valid: false,
                ..params(MatteChannelCode::Alpha)
            },
        );
        assert_eq!(missing_bypass, OVERLAY);
    }

    #[test]
    fn resource_plan_rejects_hostile_dimensions_counts_and_bytes() {
        let generous = MatteResourceLimits {
            max_texture_dimension_2d: 16_384,
            max_texture_array_layers: 256,
            max_sampled_textures_per_shader_stage: 16,
            max_bytes: MAX_IMAGE_TAP_BYTES,
        };
        assert!(MatteResourcePlan::validate([1920, 1080], 3, generous).is_ok());
        assert!(MatteResourcePlan::validate([0, 1080], 3, generous).is_err());
        assert!(MatteResourcePlan::validate([1920, 1080], 65, generous).is_err());
        assert!(MatteResourcePlan::validate(
            [1920, 1080],
            3,
            MatteResourceLimits {
                max_texture_array_layers: 2,
                ..generous
            }
        )
        .is_err());
        assert!(MatteResourcePlan::validate(
            [1920, 1080],
            3,
            MatteResourceLimits {
                max_bytes: 1024,
                ..generous
            }
        )
        .is_err());
    }

    #[test]
    fn selective_vhs_rejects_active_mattes_instead_of_silently_dropping_them() {
        assert!(validate_selective_matte_topology(false).is_ok());
        assert_eq!(
            validate_selective_matte_topology(true).unwrap_err(),
            SELECTIVE_MATTE_TOPOLOGY_ERROR
        );
    }

    fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Matte compositor test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device")
    }

    fn pixel_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        pixel: [u8; 4],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn readback_target(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
    ) -> [u8; 4] {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matte compositor pixel readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(
            std::mem::replace(
                encoder,
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Unused replacement encoder"),
                }),
            )
            .finish(),
        ));
        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        receive.recv().expect("map callback").expect("map result");
        let data = slice.get_mapped_range();
        let pixel = [data[0], data[1], data[2], data[3]];
        drop(data);
        staging.unmap();
        pixel
    }

    fn render_matte_pixel(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base: [u8; 4],
        overlay: [u8; 4],
        donor: [u8; 4],
        uniforms: MatteCompositeUniforms,
    ) -> [u8; 4] {
        let (_base_texture, base_view) = pixel_texture(device, queue, "Matte base", base);
        let (_overlay_texture, overlay_view) =
            pixel_texture(device, queue, "Matte overlay", overlay);
        let (_donor_texture, donor_view) = pixel_texture(device, queue, "Matte donor", donor);
        let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matte test vertex"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
        });
        let pipeline = MatteCompositePipeline::build(device, &vertex);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matte test uniforms"),
            size: std::mem::size_of::<MatteCompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matte test textures"),
            layout: &pipeline.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&donor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matte test uniform group"),
            layout: &pipeline.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Matte test target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matte test encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Matte test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &textures, &[]);
            pass.set_bind_group(1, &uniform_group, &[]);
            pass.draw(0..3, 0..1);
        }
        readback_target(device, queue, &mut encoder, &target)
    }

    fn render_legacy_pixel(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base: [u8; 4],
        overlay: [u8; 4],
        opacity: f32,
        blend_mode: u32,
    ) -> [u8; 4] {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct LegacyUniforms {
            opacity: f32,
            blend_mode: u32,
            pad: [f32; 2],
        }

        let (_base_texture, base_view) = pixel_texture(device, queue, "Legacy base", base);
        let (_overlay_texture, overlay_view) =
            pixel_texture(device, queue, "Legacy overlay", overlay);
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Legacy test texture BGL"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Legacy test uniform BGL"),
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
            label: Some("Legacy test vertex"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Legacy test fragment"),
            source: wgpu::ShaderSource::Wgsl(composite_shader_source()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Legacy test pipeline layout"),
            bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Legacy test pipeline"),
            layout: Some(&pipeline_layout),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Legacy test textures"),
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Legacy test uniforms"),
            size: std::mem::size_of::<LegacyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniform_buffer,
            0,
            bytemuck::bytes_of(&LegacyUniforms {
                opacity,
                blend_mode,
                pad: [0.0; 2],
            }),
        );
        let uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Legacy test uniform group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Legacy test target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Legacy test encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Legacy test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &textures, &[]);
            pass.set_bind_group(1, &uniforms, &[]);
            pass.draw(0..3, 0..1);
        }
        readback_target(device, queue, &mut encoder, &target)
    }

    fn srgb_byte_to_linear(value: u8) -> f32 {
        let encoded = f32::from(value) / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
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

    fn reference_pixel(mode: BlendMode, base: [u8; 4], overlay: [u8; 4], opacity: f32) -> [u8; 4] {
        let base_linear = [
            srgb_byte_to_linear(base[0]),
            srgb_byte_to_linear(base[1]),
            srgb_byte_to_linear(base[2]),
            f32::from(base[3]) / 255.0,
        ];
        let overlay_linear = [
            srgb_byte_to_linear(overlay[0]),
            srgb_byte_to_linear(overlay[1]),
            srgb_byte_to_linear(overlay[2]),
            f32::from(overlay[3]) / 255.0,
        ];
        let output = composite_straight(mode, base_linear, overlay_linear, opacity);
        [
            linear_to_srgb_byte(output[0]),
            linear_to_srgb_byte(output[1]),
            linear_to_srgb_byte(output[2]),
            (output[3].clamp(0.0, 1.0) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
        ]
    }

    fn pixel_close(actual: [u8; 4], expected: [u8; 4], context: &str) {
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                actual.abs_diff(expected) <= 1,
                "{context}, channel {channel}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_all_blend_modes_match_linear_cpu_opaque_transparent_and_half_alpha_vectors() {
        let (device, queue) = gpu_device();
        let cases = [
            ("opaque", [50, 145, 225, 255], [215, 95, 170, 255], 1.0),
            (
                "transparent-bottom",
                [230, 45, 155, 0],
                [75, 205, 115, 128],
                1.0,
            ),
            ("half-alpha", [80, 145, 220, 128], [210, 90, 165, 128], 0.75),
        ];

        for mode in BlendMode::ALL {
            for (case, base, overlay, opacity) in cases {
                let expected = reference_pixel(mode, base, overlay, opacity);
                let actual =
                    render_legacy_pixel(&device, &queue, base, overlay, opacity, mode.as_u32());
                pixel_close(actual, expected, &format!("{} {case}", mode.key()));
            }

            // A fully dry matte must compile and execute the same kernel for
            // every mode, not a stale four-mode copy.
            let base = [80, 145, 220, 128];
            let overlay = [210, 90, 165, 128];
            let expected = reference_pixel(mode, base, overlay, 0.75);
            let matte = render_matte_pixel(
                &device,
                &queue,
                base,
                overlay,
                [255; 4],
                MatteCompositeUniforms::new(
                    0.75,
                    mode.as_u32(),
                    ResolvedMatteParams {
                        channel: MatteChannelCode::Alpha,
                        invert: true,
                        amount: 0.0,
                        threshold: 0.5,
                        softness: 0.1,
                        donor_valid: false,
                    },
                ),
            );
            pixel_close(matte, expected, &format!("matte {} half-alpha", mode.key()));
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_disabled_matte_branch_matches_legacy_reference_arithmetic() {
        let (device, queue) = gpu_device();
        let base = [30, 70, 140, 96];
        let overlay = [200, 100, 50, 128];
        let expected = render_legacy_pixel(&device, &queue, base, overlay, 0.65, 3);
        let actual = render_matte_pixel(
            &device,
            &queue,
            base,
            overlay,
            [255; 4],
            MatteCompositeUniforms::new(
                0.65,
                3,
                ResolvedMatteParams {
                    channel: MatteChannelCode::Alpha,
                    invert: true,
                    amount: 0.0,
                    threshold: 0.5,
                    softness: 0.1,
                    donor_valid: false,
                },
            ),
        );
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_matte_half_alpha_and_missing_route_obey_straight_alpha() {
        let (device, queue) = gpu_device();
        let overlay = [200, 100, 50, 128];
        let half = render_matte_pixel(
            &device,
            &queue,
            [0; 4],
            overlay,
            [0, 0, 0, 128],
            MatteCompositeUniforms::new(
                1.0,
                0,
                ResolvedMatteParams {
                    channel: MatteChannelCode::Alpha,
                    invert: false,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 1.0,
                    donor_valid: true,
                },
            ),
        );
        for (actual, expected) in half[..3].iter().zip(overlay[..3].iter()) {
            assert!(
                actual.abs_diff(*expected) <= 1,
                "actual={actual}, expected={expected}"
            );
        }
        assert!(
            half[3].abs_diff(64) <= 1,
            "half-alpha output was {}",
            half[3]
        );

        let missing = render_matte_pixel(
            &device,
            &queue,
            [0; 4],
            overlay,
            [255; 4],
            MatteCompositeUniforms::new(
                1.0,
                0,
                ResolvedMatteParams {
                    channel: MatteChannelCode::Alpha,
                    invert: true,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 1.0,
                    donor_valid: false,
                },
            ),
        );
        assert_eq!(missing[3], 0, "invert resurrected a missing donor");
    }
}
