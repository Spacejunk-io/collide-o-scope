//! The fixed-pipeline Study interpreter executor.
//!
//! One shader (`study_interpreter.wgsl`), compiled once at construction and
//! never generated: a compiled Study arrives as a bounded uniform
//! instruction buffer and the fragment stage walks it. Swapping studies is
//! two `write_buffer` calls into fixed-size buffers — no reallocation, no
//! pipeline change, no layout change. Two sampled textures (carrier and the
//! committed clean-history D2 array), no sampler; every lookup is a
//! `textureLoad`, inside the ordinary three-texture rack ceiling.
//!
//! The executor lands with the S10b interpreter tranche one step ahead of an
//! authored audience surface — where a Study plugs into the composition
//! (rack node kind, master slot) is a product decision the operator has not
//! yet opened, exactly the browser-surface pattern. Until that tranche its
//! only callers are the CPU-agreement fixtures below, and this allow is
//! scoped to that window.
#![allow(dead_code)]

use crate::study_eval::{CompiledStudy, StudyFrameContext, StudyGpuOp, STUDY_GPU_MAX_INSTRUCTIONS};

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
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<StudyGpuFrameUniforms>() == 64);
const _: () = assert!(std::mem::size_of::<StudyGpuOp>() * STUDY_GPU_MAX_INSTRUCTIONS == 8_192);

impl StudyGpuFrameUniforms {
    /// Build the block from the same context the CPU reference consumes,
    /// applying the identical input sanitation (non-finite lands on the
    /// documented neutral, bands and phase clamp to `0..=1`) so the two
    /// halves observe the same numbers.
    pub fn from_context(
        frame: &StudyFrameContext,
        compiled: &CompiledStudy,
        write_index: u32,
        history_len: u32,
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
            instruction_count: compiled.instruction_count(),
            valid_history: frame.valid_history,
            write_index,
            history_len,
            _pad: [0; 3],
        }
    }
}

pub struct StudyGpuExecutor {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    frame_buffer: wgpu::Buffer,
    program_buffer: wgpu::Buffer,
}

impl StudyGpuExecutor {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Study interpreter bind layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2),
                texture_entry(1, wgpu::TextureViewDimension::D2Array),
                uniform_entry(2, std::mem::size_of::<StudyGpuFrameUniforms>() as u64),
                uniform_entry(
                    3,
                    (std::mem::size_of::<StudyGpuOp>() * STUDY_GPU_MAX_INSTRUCTIONS) as u64,
                ),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Study interpreter shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/study_interpreter.wgsl").into(),
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
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Study interpreter frame uniforms"),
            size: std::mem::size_of::<StudyGpuFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let program_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Study interpreter program"),
            size: (std::mem::size_of::<StudyGpuOp>() * STUDY_GPU_MAX_INSTRUCTIONS) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_layout,
            frame_buffer,
            program_buffer,
        }
    }

    /// Upload a compiled study and its frame block. Fixed-size buffers: a
    /// study swap is exactly these two writes.
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        program: &[StudyGpuOp],
        frame: &StudyGpuFrameUniforms,
    ) {
        debug_assert_eq!(program.len(), STUDY_GPU_MAX_INSTRUCTIONS);
        queue.write_buffer(&self.program_buffer, 0, bytemuck::cast_slice(program));
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(frame));
    }

    /// Bind the carrier and the committed history array. Callers cache the
    /// group per (carrier, history) view pair; a warm frame creates nothing.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        carrier: &wgpu::TextureView,
        history: &wgpu::TextureView,
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
                    resource: self.frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.program_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode exactly one fullscreen pass into `target`.
    pub fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
    ) {
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
        pass.set_bind_group(0, bind_group, &[]);
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

fn uniform_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(min_size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{StudyCapability, StudyInstruction};
    use crate::study_eval::tests::{document, every_opcode_document, register};
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
        executor.encode_pass(&mut encoder, bind_group, &view);
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
        let history = RingHistory { write_index };
        let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                out.push(compiled.evaluate_pixel(
                    frame,
                    &StudyPixelInputs {
                        current: carrier_pixel(x, y),
                        motion: [0.0, 0.0],
                        history: &history,
                    },
                ));
            }
        }
        out
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
    /// and the resolved randomness observably contributing.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_study_interpreter_matches_the_cpu_reference_across_every_opcode() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let compiled = crate::study_eval::CompiledStudy::compile(&every_opcode_document()).unwrap();
        let executor = StudyGpuExecutor::new(&device, wgpu::TextureFormat::Rgba32Float);

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

        executor.upload(
            &queue,
            &compiled.encode_gpu_program(),
            &StudyGpuFrameUniforms::from_context(&frame, &compiled, write_index, HISTORY_LEN),
        );
        let bind_group = executor.create_bind_group(&device, &carrier_view, &history_view);
        let gpu = render_and_read(&device, &queue, &executor, &bind_group);
        let cpu = cpu_reference(&compiled, &frame, write_index);
        assert_agreement(&gpu, &cpu, "every-opcode document");

        // The claims must be discriminating: a fully-committed ring changes
        // the image (the guard was doing work), and both halves track it.
        let mut committed = frame;
        committed.valid_history = HISTORY_LEN;
        executor.upload(
            &queue,
            &compiled.encode_gpu_program(),
            &StudyGpuFrameUniforms::from_context(&committed, &compiled, write_index, HISTORY_LEN),
        );
        let gpu_committed = render_and_read(&device, &queue, &executor, &bind_group);
        let cpu_committed = cpu_reference(&compiled, &committed, write_index);
        assert_agreement(&gpu_committed, &cpu_committed, "committed ring");
        assert_ne!(gpu, gpu_committed, "the history guard must be observable");
    }

    /// A study swap is two buffer writes: the same executor and the same
    /// bind group render a second document correctly, and re-rendering it is
    /// byte-identical.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_study_swap_is_two_writes_into_fixed_buffers_and_stays_deterministic() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let executor = StudyGpuExecutor::new(&device, wgpu::TextureFormat::Rgba32Float);
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

        executor.upload(
            &queue,
            &second.encode_gpu_program(),
            &StudyGpuFrameUniforms::from_context(&frame, &second, write_index, HISTORY_LEN),
        );
        let first_render = render_and_read(&device, &queue, &executor, &bind_group);
        assert_agreement(
            &first_render,
            &cpu_reference(&second, &frame, write_index),
            "swapped document",
        );
        let second_render = render_and_read(&device, &queue, &executor, &bind_group);
        assert_eq!(
            first_render, second_render,
            "re-rendering the same program must be deterministic"
        );
    }
}
