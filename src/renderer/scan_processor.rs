//! The dedicated Scan Processor executor — the tree's first
//! non-fullscreen-triangle pass.
//!
//! One shader (the canonical `blend.wgsl` kernel plus
//! `scan_processor.wgsl`), compiled once at construction and never
//! generated. Two pipelines share it: the instanced ribbon geometry pass
//! (two vertices per beam sample, one instance per scanline, no vertex
//! buffers — position from `vertex_index`/`instance_index`, carrier fetched
//! in the vertex stage through the explicit-load bilinear, sampler-free)
//! accumulating additively into one shared transient `Rgba16Float`
//! accumulator cleared to alpha one, and the fullscreen resolve applying the
//! engine-wide node wet/blend law through the one blend kernel. The vertex
//! count is a draw-call argument, so a lines or samples edit re-encodes the
//! next frame without touching pipelines, arenas, or the accumulator.

use crate::evaluated_frame::evaluated_composition::EvaluatedScanProcessorPlan;
use crate::scan_processor::ScanProcessorParams;
use crate::visual_rack::NodeBlend;

const SCAN_UNIFORM_BYTES: u64 = 128;

/// The 128-byte uniform record, mirrored lane for lane by `ScanUniforms` in
/// the shader and restated as `SCAN_PROCESSOR_UNIFORM_BYTES` in the planner
/// ledger.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScanProcessorGpuUniforms {
    /// amount, ribbon_width, velocity_mix, collapse
    pub deflect: [f32; 4],
    /// tilt_x, tilt_y, perspective, s_curve
    pub surface: [f32; 4],
    /// skew, osc_amount, osc_freq, osc_lock
    pub osc: [f32; 4],
    /// lissajous, mono, hue, time_seconds
    pub color_time: [f32; 4],
    /// lines, samples_per_line, reverse_h, reverse_v
    pub raster: [f32; 4],
    /// output width, output height, wet, reserved
    pub frame: [f32; 4],
    /// blend code, reserved x3
    pub modes: [u32; 4],
    pub reserved: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<ScanProcessorGpuUniforms>() as u64 == SCAN_UNIFORM_BYTES);

impl ScanProcessorGpuUniforms {
    /// Pack sanitized authored params plus the renderer-owned frame lanes.
    /// The only time input is the shared frame-plan seconds, never wall
    /// time, so Pause holds the detuned oscillator still and export replays
    /// it structurally.
    pub fn from_parts(
        params: &ScanProcessorParams,
        output: [u32; 2],
        time_seconds: f32,
        wet: f32,
        blend: NodeBlend,
    ) -> Self {
        let clean = params.sanitized();
        let time = if time_seconds.is_finite() {
            time_seconds
        } else {
            0.0
        };
        Self {
            deflect: [
                clean.amount,
                clean.ribbon_width,
                clean.velocity_mix,
                clean.collapse,
            ],
            surface: [clean.tilt_x, clean.tilt_y, clean.perspective, clean.s_curve],
            osc: [clean.skew, clean.osc_amount, clean.osc_freq, clean.osc_lock],
            color_time: [clean.lissajous, clean.mono, clean.hue, time],
            raster: [
                clean.lines as f32,
                clean.samples_per_line as f32,
                if clean.reverse_h { 1.0 } else { 0.0 },
                if clean.reverse_v { 1.0 } else { 0.0 },
            ],
            frame: [
                output[0] as f32,
                output[1] as f32,
                if wet.is_finite() {
                    wet.clamp(0.0, 1.0)
                } else {
                    0.0
                },
                0.0,
            ],
            modes: [blend.code(), 0, 0, 0],
            reserved: [0.0; 4],
        }
    }
}

pub struct ScanProcessorGpuExecutor {
    geometry_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    geometry_bind_layout: wgpu::BindGroupLayout,
    resolve_bind_layout: wgpu::BindGroupLayout,
    uniform_arena: wgpu::Buffer,
    uniform_stride: u32,
    slots: u32,
    /// The one shared transient the ribbons accumulate into. Allocated once
    /// at prepare, cleared per pass, reused by every scan node in the frame.
    accumulator_view: wgpu::TextureView,
}

impl ScanProcessorGpuExecutor {
    /// One inert-plan predicate for the planner-emitted step: disabled, dry,
    /// or an exact bypass (no deflection authored) encodes nothing and the
    /// carrier passes through untouched.
    pub fn is_inert(plan: &EvaluatedScanProcessorPlan) -> bool {
        !plan.enabled || plan.wet <= 0.0 || plan.params.is_exact_bypass()
    }

    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        slots: u32,
        output: [u32; 2],
    ) -> Self {
        let slots = slots.max(1);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let uniform_stride = SCAN_UNIFORM_BYTES.div_ceil(u64::from(align)) as u32 * align;
        let geometry_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scan Processor geometry bind layout"),
                entries: &[
                    // The carrier is fetched in the vertex stage — the whole
                    // point of the pass.
                    texture_entry(0, wgpu::ShaderStages::VERTEX),
                    uniform_entry(2, wgpu::ShaderStages::VERTEX),
                ],
            });
        let resolve_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scan Processor resolve bind layout"),
                entries: &[
                    texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                ],
            });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scan Processor shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/blend.wgsl"),
                    include_str!("../shaders/scan_processor.wgsl"),
                )
                .into(),
            ),
        });
        let geometry_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scan Processor geometry pipeline layout"),
            bind_group_layouts: &[Some(&geometry_bind_layout)],
            immediate_size: 0,
        });
        let geometry_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scan Processor geometry pipeline"),
            layout: Some(&geometry_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_scan"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_scan"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ACCUMULATOR_FORMAT,
                    // Where lines bunch, they add up — the whole mechanism.
                    // Contributions carry alpha zero over a clear to alpha
                    // one, so coverage cannot stack past unity.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scan Processor resolve pipeline layout"),
            bind_group_layouts: &[Some(&resolve_bind_layout)],
            immediate_size: 0,
        });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scan Processor resolve pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_scan_resolve"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_scan_resolve"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // The resolve overwrites every texel; no blend.
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
        });
        let uniform_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Processor uniform arena"),
            size: u64::from(uniform_stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accumulator = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scan Processor accumulator"),
            size: wgpu::Extent3d {
                width: output[0].max(1),
                height: output[1].max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ACCUMULATOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let accumulator_view = accumulator.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            geometry_pipeline,
            resolve_pipeline,
            geometry_bind_layout,
            resolve_bind_layout,
            uniform_arena,
            uniform_stride,
            slots,
            accumulator_view,
        }
    }

    /// Upload one slot's frame uniforms. Written once per encoded frame.
    pub fn write_frame(&self, queue: &wgpu::Queue, slot: u32, frame: &ScanProcessorGpuUniforms) {
        debug_assert!(slot < self.slots);
        queue.write_buffer(
            &self.uniform_arena,
            u64::from(slot) * u64::from(self.uniform_stride),
            bytemuck::bytes_of(frame),
        );
    }

    /// Bind the carrier for the geometry pass. Callers cache the group per
    /// carrier view; a warm frame creates nothing.
    pub fn create_geometry_bind_group(
        &self,
        device: &wgpu::Device,
        carrier: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Processor geometry bind group"),
            layout: &self.geometry_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(carrier),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_arena,
                        offset: 0,
                        size: std::num::NonZeroU64::new(SCAN_UNIFORM_BYTES),
                    }),
                },
            ],
        })
    }

    /// Bind the carrier and the shared accumulator for the resolve pass.
    pub fn create_resolve_bind_group(
        &self,
        device: &wgpu::Device,
        carrier: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Processor resolve bind group"),
            layout: &self.resolve_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(carrier),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.accumulator_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_arena,
                        offset: 0,
                        size: std::num::NonZeroU64::new(SCAN_UNIFORM_BYTES),
                    }),
                },
            ],
        })
    }

    /// Encode exactly one drawn raster: the instanced geometry pass into the
    /// cleared accumulator, then the fullscreen resolve into `target`.
    pub fn encode_at(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        geometry_bind_group: &wgpu::BindGroup,
        resolve_bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        slot: u32,
        params: &ScanProcessorParams,
    ) {
        debug_assert!(slot < self.slots);
        let clean = params.sanitized();
        let offset = slot * self.uniform_stride;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scan Processor geometry pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.accumulator_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Cleared to alpha one: the drawn raster claims full
                        // coverage, and additive contributions carry none.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.geometry_pipeline);
            pass.set_bind_group(0, geometry_bind_group, &[offset]);
            // Two vertices per sample make the ribbon; one instance per
            // scanline.
            pass.draw(0..clean.samples_per_line * 2, 0..clean.lines);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Scan Processor resolve pass"),
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
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, resolve_bind_group, &[offset]);
        pass.draw(0..3, 0..1);
    }
}

const ACCUMULATOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(SCAN_UNIFORM_BYTES),
        },
        count: None,
    }
}
