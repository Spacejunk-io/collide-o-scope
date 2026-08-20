//! The B6 corruption-trio executor: one machine, three laws.
//!
//! Four pipelines over one shader (`corruption.wgsl`, composed with the
//! canonical `blend.wgsl` kernel): the Block DCT's coefficient and final
//! reconstruction stages, the Pixel Sort pass, and the Filter Avalanche
//! pass. One dynamic-offset uniform arena serves every encoded pass, and
//! the Block DCT's two full-frame `Rgba16Float` coefficient intermediates
//! are shared by every DCT step in the frame — the Scan-accumulator law:
//! sequential reuse, charged once. Sampler-free; two bound textures per
//! pass.
//!
//! The avalanche's retained per-node history lives with the composition
//! executor (the bus-melt lazy/staged/committed shape), not here: this
//! module owns pipelines and arenas, never per-frame state.

use crate::evaluated_frame::evaluated_composition::{
    EvaluatedCorruptionKind, EvaluatedCorruptionPlan,
};

const CORRUPTION_UNIFORM_BYTES: u64 = 80;

/// Pass modes, mirrored by `corruption.wgsl`.
const MODE_DCT_COEF_CARRIER: u32 = 0;
const MODE_DCT_RECON_MID: u32 = 1;
const MODE_DCT_COEF_AUX: u32 = 2;
const MODE_DCT_FINAL: u32 = 3;
const MODE_PIXEL_SORT: u32 = 4;
const MODE_AVALANCHE: u32 = 5;

/// The per-pass uniform record, mirrored field for field by
/// `CorruptionUniforms` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CorruptionGpuUniforms {
    /// pass mode, axis, blend code, dct block edge.
    pub meta: [u32; 4],
    /// amount, quantize/threshold, hf penalty, chroma crush.
    pub params0: [f32; 4],
    /// node wet, avalanche span, spare, spare.
    pub params1: [f32; 4],
    /// master seed, avalanche epoch, history_valid, spare.
    pub meta2: [u32; 4],
    /// output dimensions.
    pub size: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<CorruptionGpuUniforms>() == 80);

impl CorruptionGpuUniforms {
    /// The records for every pass one active step encodes, in encode order.
    /// The seed/epoch pair feeds only the avalanche's deterministic lanes.
    pub fn for_plan(
        plan: &EvaluatedCorruptionPlan,
        output: [u32; 2],
        seed: u32,
        avalanche_epoch: u32,
        history_valid: bool,
    ) -> Vec<Self> {
        let base = Self {
            meta: [0, 0, plan.blend.code(), 0],
            params0: [0.0; 4],
            params1: [plan.wet.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            meta2: [seed, avalanche_epoch, u32::from(history_valid), 0],
            size: [output[0] as f32, output[1] as f32, 0.0, 0.0],
        };
        match plan.kind {
            EvaluatedCorruptionKind::BlockDct(params) => {
                let params = params.sanitized();
                let edge = params.block_edge();
                let stage = |mode: u32, axis: u32| Self {
                    meta: [mode, axis, plan.blend.code(), edge],
                    params0: [
                        params.amount,
                        params.quantize,
                        params.hf_penalty,
                        params.chroma_crush,
                    ],
                    ..base
                };
                vec![
                    stage(MODE_DCT_COEF_CARRIER, 0),
                    stage(MODE_DCT_RECON_MID, 0),
                    stage(MODE_DCT_COEF_AUX, 1),
                    stage(MODE_DCT_FINAL, 1),
                ]
            }
            EvaluatedCorruptionKind::PixelSort(params) => {
                let params = params.sanitized();
                vec![Self {
                    meta: [MODE_PIXEL_SORT, 0, plan.blend.code(), 0],
                    params0: [params.amount, params.threshold, 0.0, 0.0],
                    ..base
                }]
            }
            EvaluatedCorruptionKind::Avalanche(params) => {
                let params = params.sanitized();
                vec![Self {
                    meta: [MODE_AVALANCHE, params.axis.code(), plan.blend.code(), 0],
                    params0: [params.amount, 0.0, 0.0, 0.0],
                    params1: [plan.wet.clamp(0.0, 1.0), params.span(), 0.0, 0.0],
                    ..base
                }]
            }
        }
    }
}

/// The two shared DCT coefficient intermediates.
pub struct CorruptionAux {
    #[allow(
        dead_code,
        reason = "the texture is retained so its views stay alive; only the views are bound"
    )]
    pub textures: [wgpu::Texture; 2],
    pub views: [wgpu::TextureView; 2],
}

pub struct CorruptionGpuExecutor {
    dct_stage_pipeline: wgpu::RenderPipeline,
    dct_final_pipeline: wgpu::RenderPipeline,
    sort_pipeline: wgpu::RenderPipeline,
    avalanche_pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    arena: wgpu::Buffer,
    stride: u32,
    slots: u32,
    aux: Option<CorruptionAux>,
}

impl CorruptionGpuExecutor {
    /// One inert-plan predicate: disabled, dry, or an exact bypass encodes
    /// nothing and the carrier passes through untouched.
    pub fn is_inert(plan: &EvaluatedCorruptionPlan) -> bool {
        !plan.is_active()
    }

    /// `slots` is the total encoded-pass count across every corruption step
    /// (a DCT step owns four, the others one). `with_dct` allocates the
    /// shared intermediates only when a DCT step exists in the plan, so a
    /// DCT-free session charges no aux bytes.
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        slots: u32,
        dimensions: [u32; 2],
        with_dct: bool,
    ) -> Self {
        let slots = slots.max(1);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride =
            (CORRUPTION_UNIFORM_BYTES.div_ceil(u64::from(align)) * u64::from(align)) as u32;
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corruption bind layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(CORRUPTION_UNIFORM_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Corruption shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/blend.wgsl"),
                    include_str!("../shaders/corruption.wgsl"),
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corruption pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let make_pipeline = |label: &str, entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_corruption"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let dct_stage_pipeline = make_pipeline(
            "Corruption DCT stage pipeline",
            "fs_dct_stage",
            wgpu::TextureFormat::Rgba16Float,
        );
        let dct_final_pipeline = make_pipeline(
            "Corruption DCT final pipeline",
            "fs_dct_final",
            target_format,
        );
        let sort_pipeline = make_pipeline(
            "Corruption pixel sort pipeline",
            "fs_pixel_sort",
            target_format,
        );
        let avalanche_pipeline = make_pipeline(
            "Corruption avalanche pipeline",
            "fs_avalanche",
            target_format,
        );
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corruption uniform arena"),
            size: u64::from(stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aux = with_dct.then(|| {
            let make = |index: usize| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Corruption DCT intermediate {index}")),
                    size: wgpu::Extent3d {
                        width: dimensions[0].max(1),
                        height: dimensions[1].max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            };
            let textures = [make(0), make(1)];
            let views = [
                textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
                textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
            ];
            CorruptionAux { textures, views }
        });
        Self {
            dct_stage_pipeline,
            dct_final_pipeline,
            sort_pipeline,
            avalanche_pipeline,
            bind_layout,
            arena,
            stride,
            slots,
            aux,
        }
    }

    pub fn aux_views(&self) -> Option<[&wgpu::TextureView; 2]> {
        self.aux.as_ref().map(|aux| [&aux.views[0], &aux.views[1]])
    }

    /// Upload one pass slot's record.
    pub fn write_pass(&self, queue: &wgpu::Queue, slot: u32, uniforms: &CorruptionGpuUniforms) {
        debug_assert!(slot < self.slots);
        queue.write_buffer(
            &self.arena,
            u64::from(slot) * u64::from(self.stride),
            bytemuck::bytes_of(uniforms),
        );
    }

    /// Bind one pass's texture pair. Callers prebuild the groups at prepare
    /// (and at the avalanche history's lazy allocation); a warm frame
    /// creates nothing.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        primary: &wgpu::TextureView,
        secondary: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corruption bind group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(primary),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(secondary),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.arena,
                        offset: 0,
                        size: std::num::NonZeroU64::new(CORRUPTION_UNIFORM_BYTES),
                    }),
                },
            ],
        })
    }

    /// Encode one fullscreen pass for `slot` into `target` with the
    /// pipeline the pass mode selects.
    pub fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mode: CorruptionPassPipeline,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        slot: u32,
    ) {
        debug_assert!(slot < self.slots);
        let pipeline = match mode {
            CorruptionPassPipeline::DctStage => &self.dct_stage_pipeline,
            CorruptionPassPipeline::DctFinal => &self.dct_final_pipeline,
            CorruptionPassPipeline::PixelSort => &self.sort_pipeline,
            CorruptionPassPipeline::Avalanche => &self.avalanche_pipeline,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Corruption pass"),
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
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[slot * self.stride]);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionPassPipeline {
    DctStage,
    DctFinal,
    PixelSort,
    Avalanche,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual_rack::{NodeBlend, NodeId};

    const SIZE: u32 = 64;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Corruption test"),
            ..Default::default()
        }))
        .ok()
    }

    /// A deterministic synthetic image in the encoded domain: gradients, a
    /// hard bright block, and a dark band, so every law sees runs, edges,
    /// and flats.
    fn synthetic_encoded() -> Vec<[f32; 3]> {
        let mut image = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let mut px = [
                    x as f32 / (SIZE - 1) as f32,
                    y as f32 / (SIZE - 1) as f32,
                    0.25,
                ];
                if x > 20 && x < 44 && y > 8 && y < 40 {
                    px = [0.9, 0.85, 0.8];
                }
                if y > 52 {
                    px = [0.05, 0.08, 0.06];
                }
                image.push(px);
            }
        }
        image
    }

    fn encoded_to_linear(value: f32) -> f32 {
        let value = value.clamp(0.0, 1.0);
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_encoded(value: f32) -> f32 {
        let value = value.clamp(0.0, 1.0);
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    }

    fn upload_linear(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoded: &[[f32; 3]],
    ) -> wgpu::Texture {
        let mut bytes = Vec::with_capacity(encoded.len() * 16);
        for px in encoded {
            for channel in px {
                bytes.extend_from_slice(&encoded_to_linear(*channel).to_le_bytes());
            }
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corruption fixture source"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
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
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 16),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        texture
    }

    fn read_encoded(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<[f32; 3]> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corruption fixture readback"),
            size: u64::from(SIZE) * u64::from(SIZE) * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 16),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let mapped = buffer.slice(..).get_mapped_range();
        let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
        for px in mapped.chunks_exact(16) {
            let read =
                |offset: usize| f32::from_le_bytes(px[offset..offset + 4].try_into().unwrap());
            out.push([
                linear_to_encoded(read(0)),
                linear_to_encoded(read(4)),
                linear_to_encoded(read(8)),
            ]);
        }
        out
    }

    fn agreement(gpu: &[[f32; 3]], cpu: &[[f32; 3]], label: &str) {
        let mut within = 0usize;
        let mut total = 0usize;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            for channel in 0..3 {
                total += 1;
                if (a[channel] - b[channel]).abs() <= 4.0 / 255.0 {
                    within += 1;
                }
            }
        }
        let fraction = within as f64 / total as f64;
        assert!(
            fraction >= 0.95,
            "{label}: GPU/CPU agreement {fraction:.3} below the B7 statistical contract"
        );
    }

    fn plan_for(kind: EvaluatedCorruptionKind) -> EvaluatedCorruptionPlan {
        EvaluatedCorruptionPlan {
            node_id: NodeId::new(9).unwrap(),
            enabled: true,
            wet: 1.0,
            blend: NodeBlend::Normal,
            kind,
            resources: Default::default(),
        }
    }

    fn make_target(device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corruption fixture target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// The three GPU laws against their CPU references, to the B7
    /// statistical contract (the DCT's f16 coefficient intermediates and
    /// per-adapter transcendentals move isolated samples; a wrong law moves
    /// most of them and fails flat). The avalanche is additionally proven to
    /// read its history binding: a warm history changes the picture exactly
    /// as the CPU reference says it must.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_corruption_trio_matches_the_cpu_references() {
        use crate::block_dct::{block_dct_line, BlockDctParams};
        use crate::filter_avalanche::{avalanche_reference, AvalancheAxis, AvalancheParams};
        use crate::pixel_sort::{pixel_sort_reference, PixelSortParams};

        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let encoded = synthetic_encoded();
        let source = upload_linear(&device, &queue, &encoded);
        let source_view = source.create_view(&Default::default());
        let executor = CorruptionGpuExecutor::new(
            &device,
            wgpu::TextureFormat::Rgba32Float,
            8,
            [SIZE, SIZE],
            true,
        );
        let target = make_target(&device);
        let target_view = target.create_view(&Default::default());

        // Pixel sort.
        let sort_params = PixelSortParams {
            amount: 1.0,
            threshold: 0.45,
        };
        let plan = plan_for(EvaluatedCorruptionKind::PixelSort(sort_params));
        let uniforms = CorruptionGpuUniforms::for_plan(&plan, [SIZE, SIZE], 9, 0, false);
        executor.write_pass(&queue, 0, &uniforms[0]);
        let group = executor.create_bind_group(&device, &source_view, &source_view);
        let mut encoder = device.create_command_encoder(&Default::default());
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::PixelSort,
            &group,
            &target_view,
            0,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let gpu = read_encoded(&device, &queue, &target);
        let cpu = pixel_sort_reference(&encoded, SIZE as usize, SIZE as usize, sort_params);
        agreement(&gpu, &cpu, "pixel sort");

        // Filter avalanche, cold (history = carrier) and warm (a distinct
        // history the cascade must inherit from).
        let avalanche_params = AvalancheParams {
            amount: 0.9,
            run: 0.6,
            axis: AvalancheAxis::Sub,
        };
        let plan = plan_for(EvaluatedCorruptionKind::Avalanche(avalanche_params));
        let seed = plan.node_id.get() as u32;
        let uniforms = CorruptionGpuUniforms::for_plan(&plan, [SIZE, SIZE], seed, 3, false);
        executor.write_pass(&queue, 1, &uniforms[0]);
        let mut encoder = device.create_command_encoder(&Default::default());
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::Avalanche,
            &group,
            &target_view,
            1,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let gpu_cold = read_encoded(&device, &queue, &target);
        // Epoch 3 spans seconds [1.0, 4/3); any timestamp inside maps
        // identically on both halves.
        let cpu_cold = avalanche_reference(
            &encoded,
            None,
            SIZE as usize,
            SIZE as usize,
            avalanche_params,
            seed,
            1.1,
        );
        agreement(&gpu_cold, &cpu_cold, "avalanche cold");

        let history_encoded: Vec<[f32; 3]> =
            encoded.iter().map(|px| [px[2], px[0], px[1]]).collect();
        let history = upload_linear(&device, &queue, &history_encoded);
        let history_view = history.create_view(&Default::default());
        let warm_group = executor.create_bind_group(&device, &source_view, &history_view);
        let warm_uniforms = CorruptionGpuUniforms::for_plan(&plan, [SIZE, SIZE], seed, 3, true);
        executor.write_pass(&queue, 2, &warm_uniforms[0]);
        let mut encoder = device.create_command_encoder(&Default::default());
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::Avalanche,
            &warm_group,
            &target_view,
            2,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let gpu_warm = read_encoded(&device, &queue, &target);
        let cpu_warm = avalanche_reference(
            &encoded,
            Some(&history_encoded),
            SIZE as usize,
            SIZE as usize,
            avalanche_params,
            seed,
            1.1,
        );
        agreement(&gpu_warm, &cpu_warm, "avalanche warm");
        assert_ne!(
            gpu_cold, gpu_warm,
            "the history binding must participate in the cascade"
        );

        // Block DCT: four GPU passes against the separable CPU law (rows,
        // then columns). At amount 1 the reconstruction replaces the
        // carrier outright.
        let dct_params = BlockDctParams {
            amount: 1.0,
            quantize: 0.5,
            hf_penalty: 0.6,
            chroma_crush: 0.7,
            block: 0.35,
        };
        let plan = plan_for(EvaluatedCorruptionKind::BlockDct(dct_params));
        let uniforms = CorruptionGpuUniforms::for_plan(&plan, [SIZE, SIZE], 9, 0, false);
        assert_eq!(uniforms.len(), 4);
        for (offset, record) in uniforms.iter().enumerate() {
            executor.write_pass(&queue, 3 + offset as u32, record);
        }
        let aux = executor.aux_views().expect("with_dct allocated the pair");
        let groups = [
            executor.create_bind_group(&device, &source_view, &source_view),
            executor.create_bind_group(&device, aux[0], &source_view),
            executor.create_bind_group(&device, aux[1], &source_view),
            executor.create_bind_group(&device, aux[0], &source_view),
        ];
        let mut encoder = device.create_command_encoder(&Default::default());
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::DctStage,
            &groups[0],
            aux[0],
            3,
        );
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::DctStage,
            &groups[1],
            aux[1],
            4,
        );
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::DctStage,
            &groups[2],
            aux[0],
            5,
        );
        executor.encode_pass(
            &mut encoder,
            CorruptionPassPipeline::DctFinal,
            &groups[3],
            &target_view,
            6,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let gpu_dct = read_encoded(&device, &queue, &target);
        let mut cpu_dct = encoded.clone();
        for y in 0..SIZE as usize {
            let row: Vec<[f32; 3]> = cpu_dct[y * SIZE as usize..(y + 1) * SIZE as usize].to_vec();
            let out = block_dct_line(&row, dct_params);
            cpu_dct[y * SIZE as usize..(y + 1) * SIZE as usize].copy_from_slice(&out);
        }
        for x in 0..SIZE as usize {
            let column: Vec<[f32; 3]> = (0..SIZE as usize)
                .map(|y| cpu_dct[y * SIZE as usize + x])
                .collect();
            let out = block_dct_line(&column, dct_params);
            for (y, px) in out.into_iter().enumerate() {
                cpu_dct[y * SIZE as usize + x] = px;
            }
        }
        agreement(&gpu_dct, &cpu_dct, "block dct");
    }
}
