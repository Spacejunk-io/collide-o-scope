//! The fixed-pipeline Study interpreter executor.
//!
//! One shader (the canonical `blend.wgsl` kernel plus
//! `study_interpreter.wgsl`), compiled once at construction and never
//! generated: a compiled Study arrives as a bounded uniform instruction
//! buffer and the fragment stage walks it. Swapping studies is two
//! `write_buffer` calls into fixed-stride arena slots — no reallocation, no
//! pipeline change, no layout change. Three sampled textures (carrier, the
//! committed clean-history D2 array, and the scope's primitive motion field),
//! no sampler; every lookup is a `textureLoad`, inside the dedicated-pass
//! ceiling. The final
//! wet/blend law is the engine-wide `apply_node_law` shape, byte-shared
//! through the one blend kernel.

use crate::evaluated_frame::evaluated_composition::EvaluatedStudyPlan;
#[cfg(test)]
use crate::study_eval::CompiledStudy;
use crate::study_eval::{StudyFrameContext, StudyGpuOp, STUDY_GPU_MAX_INSTRUCTIONS};
use crate::visual_rack::NodeBlend;

const STUDY_FRAME_UNIFORM_BYTES: u64 = 64;
const STUDY_PROGRAM_UNIFORM_BYTES: u64 =
    (std::mem::size_of::<StudyGpuOp>() * STUDY_GPU_MAX_INSTRUCTIONS) as u64;

/// The frame uniform block, mirrored field for field by
/// `StudyFrameUniforms` in the shader (bands as two vec4s).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StudyGpuFrameUniforms {
    pub audio_bands: [f32; 8],
    pub beat_phase: f32,
    pub instruction_count: u32,
    pub valid_history: u32,
    pub write_index: u32,
    pub history_len: u32,
    /// Renderer-owned node wet, applied by the engine-wide node law after
    /// the interpreter output — never an authored Study value.
    pub wet: f32,
    /// The node's frozen `NodeBlend::code()`.
    pub blend_mode: u32,
    /// ABI 1.1 motion availability for this exact committed field parity.
    pub motion_valid: u32,
}

const _: () = assert!(std::mem::size_of::<StudyGpuFrameUniforms>() == 64);
const _: () = assert!(std::mem::size_of::<StudyGpuOp>() * STUDY_GPU_MAX_INSTRUCTIONS == 8_192);

impl StudyGpuFrameUniforms {
    /// Build the block from the same context the CPU reference consumes,
    /// applying the identical input sanitation (non-finite lands on the
    /// documented neutral, bands and phase clamp to `0..=1`) so the two
    /// halves observe the same numbers. Wet 1 and Normal blend make the node
    /// law the identity over the interpreter output, which is what the
    /// CPU-agreement fixtures compare against.
    #[cfg(test)]
    pub fn from_context(
        frame: &StudyFrameContext,
        compiled: &CompiledStudy,
        write_index: u32,
        history_len: u32,
    ) -> Self {
        Self::from_parts(
            frame,
            compiled.instruction_count(),
            write_index,
            history_len,
            1.0,
            NodeBlend::Normal,
        )
    }

    pub fn from_parts(
        frame: &StudyFrameContext,
        instruction_count: u32,
        write_index: u32,
        history_len: u32,
        wet: f32,
        blend: NodeBlend,
    ) -> Self {
        let sanitize = |value: f32| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        Self {
            audio_bands: frame.audio_bands.map(sanitize),
            beat_phase: sanitize(frame.beat_phase),
            instruction_count,
            valid_history: frame.valid_history,
            write_index,
            history_len,
            wet: sanitize(wet),
            blend_mode: blend.code(),
            motion_valid: 0,
        }
    }

    pub const fn with_motion_valid(mut self, valid: bool) -> Self {
        self.motion_valid = valid as u32;
        self
    }
}

pub struct StudyGpuExecutor {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    frame_arena: wgpu::Buffer,
    program_arena: wgpu::Buffer,
    frame_stride: u32,
    program_stride: u32,
    slots: u32,
}

impl StudyGpuExecutor {
    /// One inert-plan predicate for the planner-emitted step: disabled, dry,
    /// or unresolved (no digest, or a digest the library did not hold at
    /// plan time) encodes nothing and the carrier passes through untouched.
    pub fn is_inert(plan: &EvaluatedStudyPlan) -> bool {
        !plan.enabled || plan.wet <= 0.0 || plan.program.is_none()
    }

    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, slots: u32) -> Self {
        let slots = slots.max(1);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let align_up = |value: u64| -> u64 { value.div_ceil(u64::from(align)) * u64::from(align) };
        let frame_stride = align_up(STUDY_FRAME_UNIFORM_BYTES) as u32;
        let program_stride = align_up(STUDY_PROGRAM_UNIFORM_BYTES) as u32;
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Study interpreter bind layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2),
                texture_entry(1, wgpu::TextureViewDimension::D2Array),
                texture_entry(2, wgpu::TextureViewDimension::D2),
                uniform_entry(3, STUDY_FRAME_UNIFORM_BYTES, true),
                uniform_entry(4, STUDY_PROGRAM_UNIFORM_BYTES, true),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Study interpreter shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/blend.wgsl"),
                    include_str!("../shaders/study_interpreter.wgsl"),
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Study interpreter pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Study interpreter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // The interpreter overwrites every texel; no blend, so
                    // non-blendable float targets are first-class.
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
        let frame_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Study interpreter frame arena"),
            size: u64::from(frame_stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let program_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Study interpreter program arena"),
            size: u64::from(program_stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_layout,
            frame_arena,
            program_arena,
            frame_stride,
            program_stride,
            slots,
        }
    }

    /// Upload one slot's encoded program. Topology-fixed: written at
    /// prepare, untouched per frame.
    pub fn write_program(&self, queue: &wgpu::Queue, slot: u32, program: &[StudyGpuOp]) {
        debug_assert!(slot < self.slots);
        debug_assert_eq!(program.len(), STUDY_GPU_MAX_INSTRUCTIONS);
        queue.write_buffer(
            &self.program_arena,
            u64::from(slot) * u64::from(self.program_stride),
            bytemuck::cast_slice(program),
        );
    }

    /// Upload one slot's frame block. Written once per encoded frame.
    pub fn write_frame(&self, queue: &wgpu::Queue, slot: u32, frame: &StudyGpuFrameUniforms) {
        debug_assert!(slot < self.slots);
        queue.write_buffer(
            &self.frame_arena,
            u64::from(slot) * u64::from(self.frame_stride),
            bytemuck::bytes_of(frame),
        );
    }

    /// Bind the carrier and committed history with a defined-neutral motion
    /// alias. Callers with an admitted motion field use the explicit variant.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "ABI 1.0 embedding adapters retain this neutral-motion convenience seam"
        )
    )]
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        carrier: &wgpu::TextureView,
        history: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.create_bind_group_with_motion(device, carrier, history, carrier)
    }

    pub fn create_bind_group_with_motion(
        &self,
        device: &wgpu::Device,
        carrier: &wgpu::TextureView,
        history: &wgpu::TextureView,
        motion: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Study interpreter bind group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(carrier),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(motion),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.frame_arena,
                        offset: 0,
                        size: std::num::NonZeroU64::new(STUDY_FRAME_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.program_arena,
                        offset: 0,
                        size: std::num::NonZeroU64::new(STUDY_PROGRAM_UNIFORM_BYTES),
                    }),
                },
            ],
        })
    }

    /// Encode exactly one fullscreen pass for `slot` into `target`.
    pub fn encode_at(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        slot: u32,
    ) {
        debug_assert!(slot < self.slots);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Study interpreter pass"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            bind_group,
            &[slot * self.frame_stride, slot * self.program_stride],
        );
        pass.draw(0..3, 0..1);
    }
}

fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, min_size: u64, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: std::num::NonZeroU64::new(min_size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{StudyCapability, StudyInstruction};
    use crate::study_eval::tests::{
        abi_1_1_motion_document, document, every_opcode_document, register,
    };
    use crate::study_eval::{StudyHistorySource, StudyPixelInputs};

    const SIZE: u32 = 32;
    const HISTORY_LEN: u32 = 24;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Study Interpreter Test"),
            ..Default::default()
        }))
        .ok()
    }

    fn carrier_pixel(x: u32, y: u32) -> [f32; 4] {
        [
            x as f32 / (SIZE - 1) as f32,
            y as f32 / (SIZE - 1) as f32,
            (x + y) as f32 / (2 * (SIZE - 1)) as f32,
            1.0,
        ]
    }

    fn history_layer_color(layer: u32) -> [f32; 4] {
        [
            (layer as f32 + 1.0) / 32.0,
            1.0 - (layer as f32 / 32.0),
            (layer as f32 * 7.0 % 24.0) / 24.0,
            1.0,
        ]
    }

    fn float_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: u32,
        pixel: impl Fn(u32, u32, u32) -> [f32; 4],
    ) -> wgpu::Texture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Study fixture texture"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut data = Vec::with_capacity((SIZE * SIZE * layers * 4) as usize);
        for layer in 0..layers {
            for y in 0..SIZE {
                for x in 0..SIZE {
                    data.extend_from_slice(&pixel(layer, x, y));
                }
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 16),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: layers,
            },
        );
        texture
    }

    fn render_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        executor: &StudyGpuExecutor,
        bind_group: &wgpu::BindGroup,
        slot: u32,
    ) -> Vec<[f32; 4]> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Study fixture target"),
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
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Study fixture staging"),
            size: u64::from(SIZE * SIZE * 16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        executor.encode_at(&mut encoder, bind_group, &view, slot);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
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
        let pixels = bytemuck::cast_slice::<u8, f32>(&mapped)
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        drop(mapped);
        staging.unmap();
        pixels
    }

    struct RingHistory {
        write_index: u32,
    }
    impl StudyHistorySource for RingHistory {
        fn history_color(&self, age: u8) -> [f32; 4] {
            let layer = (self.write_index + HISTORY_LEN - u32::from(age)) % HISTORY_LEN;
            history_layer_color(layer)
        }
    }

    fn cpu_reference(
        compiled: &crate::study_eval::CompiledStudy,
        frame: &crate::study_eval::StudyFrameContext,
        write_index: u32,
    ) -> Vec<[f32; 4]> {
        cpu_reference_with_motion(compiled, frame, write_index, |_, _| [0.0, 0.0])
    }

    fn cpu_reference_with_motion(
        compiled: &crate::study_eval::CompiledStudy,
        frame: &crate::study_eval::StudyFrameContext,
        write_index: u32,
        motion: impl Fn(u32, u32) -> [f32; 2],
    ) -> Vec<[f32; 4]> {
        let history = RingHistory { write_index };
        let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                out.push(compiled.evaluate_pixel(
                    frame,
                    &StudyPixelInputs {
                        current: carrier_pixel(x, y),
                        motion: motion(x, y),
                        history: &history,
                    },
                ));
            }
        }
        out
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_study_abi_1_1_motion_reaches_output_and_absence_is_neutral() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let compiled = crate::study_eval::CompiledStudy::compile(&abi_1_1_motion_document())
            .expect("ABI 1.1 motion document");
        let executor = StudyGpuExecutor::new(&device, wgpu::TextureFormat::Rgba32Float, 1);
        let carrier = float_texture(&device, &queue, 1, |_, x, y| carrier_pixel(x, y));
        let history = float_texture(&device, &queue, HISTORY_LEN, |layer, _, _| {
            history_layer_color(layer)
        });
        let carrier_view = carrier.create_view(&wgpu::TextureViewDescriptor::default());
        let history_view = history.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let frame = crate::study_eval::StudyFrameContext::default();
        executor.write_program(&queue, 0, &compiled.encode_gpu_program());

        for (label, vector) in [
            ("zero", [0.0, 0.0]),
            ("x axis", [1.0, 0.0]),
            ("y axis", [0.0, 1.0]),
            ("diagonal", [3.0, 4.0]),
            ("maximum", [65_504.0, 65_504.0]),
            ("hostile", [f32::NAN, f32::INFINITY]),
        ] {
            let motion = float_texture(&device, &queue, 1, |_, _, _| {
                [vector[0], vector[1], 0.0, 0.0]
            });
            let motion_view = motion.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = executor.create_bind_group_with_motion(
                &device,
                &carrier_view,
                &history_view,
                &motion_view,
            );
            executor.write_frame(
                &queue,
                0,
                &StudyGpuFrameUniforms::from_context(&frame, &compiled, 0, HISTORY_LEN)
                    .with_motion_valid(true),
            );
            let gpu = render_and_read(&device, &queue, &executor, &bind_group, 0);
            let cpu = cpu_reference_with_motion(&compiled, &frame, 0, |_, _| vector);
            assert_agreement(&gpu, &cpu, label);
        }

        let motion = float_texture(&device, &queue, 1, |_, _, _| [3.0, 4.0, 0.0, 0.0]);
        let motion_view = motion.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = executor.create_bind_group_with_motion(
            &device,
            &carrier_view,
            &history_view,
            &motion_view,
        );
        executor.write_frame(
            &queue,
            0,
            &StudyGpuFrameUniforms::from_context(&frame, &compiled, 0, HISTORY_LEN),
        );
        let absent = render_and_read(&device, &queue, &executor, &bind_group, 0);
        assert!(absent.iter().all(|pixel| *pixel == [0.0, 0.0, 0.0, 1.0]));
    }

    fn assert_agreement(gpu: &[[f32; 4]], cpu: &[[f32; 4]], label: &str) {
        assert_eq!(gpu.len(), cpu.len());
        for (index, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            for channel in 0..4 {
                assert!(
                    (g[channel] - c[channel]).abs() < 2.0e-5,
                    "{label}: pixel {index} channel {channel}: gpu {} vs cpu {}",
                    g[channel],
                    c[channel]
                );
            }
        }
    }

    /// The S10b agreement claim: the fixed interpreter renders the
    /// every-opcode document pixel-identical (2e-5) to the CPU reference,
    /// with the R1 guard clamping a deep history age against a young ring
    /// and the resolved randomness observably contributing. Wet 1 with
    /// Normal blend makes the node law the identity, so the comparison is
    /// against the pure interpreter output.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_study_interpreter_matches_the_cpu_reference_across_every_opcode() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let compiled = crate::study_eval::CompiledStudy::compile(&every_opcode_document()).unwrap();
        let executor = StudyGpuExecutor::new(&device, wgpu::TextureFormat::Rgba32Float, 1);

        let carrier = float_texture(&device, &queue, 1, |_, x, y| carrier_pixel(x, y));
        let history = float_texture(&device, &queue, HISTORY_LEN, |layer, _, _| {
            history_layer_color(layer)
        });
        let carrier_view = carrier.create_view(&wgpu::TextureViewDescriptor::default());
        let history_view = history.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let mut frame = crate::study_eval::StudyFrameContext::default();
        frame.audio_bands[5] = 0.66;
        frame.beat_phase = 0.33;
        // Young ring: the document's age-7 read must clamp to depth 4.
        frame.valid_history = 5;
        let write_index = 17;

        executor.write_program(&queue, 0, &compiled.encode_gpu_program());
        executor.write_frame(
            &queue,
            0,
            &StudyGpuFrameUniforms::from_context(&frame, &compiled, write_index, HISTORY_LEN),
        );
        let bind_group = executor.create_bind_group(&device, &carrier_view, &history_view);
        let gpu = render_and_read(&device, &queue, &executor, &bind_group, 0);
        let cpu = cpu_reference(&compiled, &frame, write_index);
        assert_agreement(&gpu, &cpu, "every-opcode document");

        // The claims must be discriminating: a fully-committed ring changes
        // the image (the guard was doing work), and both halves track it.
        let mut committed = frame;
        committed.valid_history = HISTORY_LEN;
        executor.write_frame(
            &queue,
            0,
            &StudyGpuFrameUniforms::from_context(&committed, &compiled, write_index, HISTORY_LEN),
        );
        let gpu_committed = render_and_read(&device, &queue, &executor, &bind_group, 0);
        let cpu_committed = cpu_reference(&compiled, &committed, write_index);
        assert_agreement(&gpu_committed, &cpu_committed, "committed ring");
        assert_ne!(gpu, gpu_committed, "the history guard must be observable");
    }

    /// A study swap is two buffer writes: the same executor and the same
    /// bind group render a second document correctly in a second arena slot,
    /// and re-rendering it is byte-identical. The wet law is also proven
    /// against the CPU-composed mix at wet 0.5.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_study_swap_is_two_writes_into_fixed_buffers_and_stays_deterministic() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let executor = StudyGpuExecutor::new(&device, wgpu::TextureFormat::Rgba32Float, 2);
        let carrier = float_texture(&device, &queue, 1, |_, x, y| carrier_pixel(x, y));
        let history = float_texture(&device, &queue, HISTORY_LEN, |layer, _, _| {
            history_layer_color(layer)
        });
        let carrier_view = carrier.create_view(&wgpu::TextureViewDescriptor::default());
        let history_view = history.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = executor.create_bind_group(&device, &carrier_view, &history_view);

        let second = crate::study_eval::CompiledStudy::compile(&document(
            vec![
                StudyCapability::CurrentColor,
                StudyCapability::HistoryRead,
                StudyCapability::AudioFeatures,
                StudyCapability::DeterministicRandom,
            ],
            vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::LoadHistoryColor {
                    dst: register(1),
                    age: 2,
                },
                StudyInstruction::LoadAudioBand {
                    dst: register(2),
                    band: 0,
                },
                StudyInstruction::LoadDeterministicRandom {
                    dst: register(3),
                    domain: 41,
                },
                StudyInstruction::Mix {
                    dst: register(4),
                    a: register(0),
                    b: register(1),
                    amount: register(2),
                },
                StudyInstruction::HueRotate {
                    dst: register(5),
                    color: register(4),
                    turns: register(3),
                },
                StudyInstruction::OutputColor { color: register(5) },
            ],
        ))
        .unwrap();

        let mut frame = crate::study_eval::StudyFrameContext::default();
        frame.audio_bands[0] = 0.4;
        frame.valid_history = HISTORY_LEN;
        let write_index = 3;

        executor.write_program(&queue, 1, &second.encode_gpu_program());
        executor.write_frame(
            &queue,
            1,
            &StudyGpuFrameUniforms::from_context(&frame, &second, write_index, HISTORY_LEN),
        );
        let first_render = render_and_read(&device, &queue, &executor, &bind_group, 1);
        assert_agreement(
            &first_render,
            &cpu_reference(&second, &frame, write_index),
            "swapped document",
        );
        let second_render = render_and_read(&device, &queue, &executor, &bind_group, 1);
        assert_eq!(
            first_render, second_render,
            "re-rendering the same program must be deterministic"
        );

        // Wet 0.5 with Normal blend must land exactly on the engine node
        // law's straight-alpha mix of carrier and study output.
        executor.write_frame(
            &queue,
            1,
            &StudyGpuFrameUniforms::from_parts(
                &frame,
                second.instruction_count(),
                write_index,
                HISTORY_LEN,
                0.5,
                NodeBlend::Normal,
            ),
        );
        let half_wet = render_and_read(&device, &queue, &executor, &bind_group, 1);
        let dry: Vec<[f32; 4]> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| carrier_pixel(x, y)))
            .collect();
        let study = cpu_reference(&second, &frame, write_index);
        for (index, observed) in half_wet.iter().enumerate() {
            let d = dry[index];
            let s = study[index];
            // apply_node_law at Normal blend: processed replaces, then the
            // wet mix interpolates premultiplied color and alpha.
            let alpha = d[3].clamp(0.0, 1.0) * 0.5 + s[3].clamp(0.0, 1.0) * 0.5;
            let expected: [f32; 4] = std::array::from_fn(|channel| {
                if channel == 3 {
                    alpha
                } else {
                    let premultiplied = d[channel].clamp(0.0, 1.0) * d[3].clamp(0.0, 1.0) * 0.5
                        + s[channel] * s[3] * 0.5;
                    if alpha <= 1.0e-6 {
                        0.0
                    } else {
                        premultiplied / alpha
                    }
                }
            });
            for channel in 0..4 {
                assert!(
                    (observed[channel] - expected[channel]).abs() < 2.0e-5,
                    "wet law: pixel {index} channel {channel}: gpu {} vs cpu {}",
                    observed[channel],
                    expected[channel]
                );
            }
        }
    }
}
