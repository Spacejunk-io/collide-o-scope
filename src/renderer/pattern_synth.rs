//! The B7 pattern-synth source executor: one fullscreen pass per pattern
//! layer per frame, rendering the layer's whole picture into its own source
//! texture before any layer-local effects pass reads it. Live and export
//! each own one instance and encode into their ordinary frame encoders, so
//! the pass runs identically on the LegacyExact path, the Advanced path,
//! and offline — there is no export-only synth path.
//!
//! The shader (`pattern_synth.wgsl`) follows `pattern_synth.rs` expression
//! for expression; its only inputs are the 128-byte uniform (authored
//! values, frame-plan time, the fixed page aspect). No texture, no sampler.
//! The executor is lazily constructed on the first pattern layer, so a
//! session that never authors one charges nothing.

use crate::pattern_synth::PatternSynthGpuUniforms;

const PATTERN_UNIFORM_BYTES: u64 = 128;

pub struct PatternSynthGpu {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    uniform_stride: u32,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u32,
}

impl PatternSynthGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pattern synth bind layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(PATTERN_UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pattern synth shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pattern_synth.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pattern synth pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pattern synth pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_pattern"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_pattern"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        let uniform_stride = uniform_stride(device);
        let capacity = 4;
        let (uniform, bind_group) = create_arena(device, &bind_layout, uniform_stride, capacity);
        Self {
            pipeline,
            bind_layout,
            uniform_stride,
            uniform,
            bind_group,
            capacity,
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, slots: u32) {
        if slots <= self.capacity {
            return;
        }
        let capacity = slots.next_power_of_two();
        let (uniform, bind_group) =
            create_arena(device, &self.bind_layout, self.uniform_stride, capacity);
        self.uniform = uniform;
        self.bind_group = bind_group;
        self.capacity = capacity;
    }

    /// Encode every pattern layer's pass into the frame encoder, one job per
    /// pattern layer, before any consumer samples the layer textures. Each
    /// job's uniforms are frame-local evaluated data (base values plus this
    /// frame's modulation offsets), never authored state read back later.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        jobs: &[(PatternSynthGpuUniforms, &wgpu::TextureView)],
    ) {
        if jobs.is_empty() {
            return;
        }
        self.ensure_capacity(device, jobs.len() as u32);
        for (slot, (uniforms, _)) in jobs.iter().enumerate() {
            queue.write_buffer(
                &self.uniform,
                u64::from(self.uniform_stride) * slot as u64,
                bytemuck::bytes_of(uniforms),
            );
        }
        for (slot, (_, target)) in jobs.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pattern synth pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Every pixel is computed; the clear only defines the
                        // load state.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[self.uniform_stride * slot as u32]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn uniform_stride(device: &wgpu::Device) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
    (PATTERN_UNIFORM_BYTES as u32).div_ceil(alignment) * alignment
}

fn create_arena(
    device: &wgpu::Device,
    bind_layout: &wgpu::BindGroupLayout,
    stride: u32,
    capacity: u32,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pattern synth uniforms"),
        size: u64::from(stride) * u64::from(capacity),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pattern synth bind group"),
        layout: bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform,
                offset: 0,
                size: std::num::NonZeroU64::new(PATTERN_UNIFORM_BYTES),
            }),
        }],
    });
    (uniform, bind_group)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_synth::{
        pattern_synth_pixel, PatternColorMode, PatternShape, PatternSynthParams,
        PATTERN_SYNTH_HEIGHT, PATTERN_SYNTH_WIDTH,
    };

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Pattern synth test"),
            ..Default::default()
        }))
        .ok()
    }

    fn read_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<[u8; 4]> {
        let padded_row = (width * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pattern fixture staging"),
            size: u64::from(padded_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
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
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            let row = &mapped[(y * padded_row) as usize..];
            for x in 0..width {
                let offset = (x * 4) as usize;
                pixels.push([
                    row[offset],
                    row[offset + 1],
                    row[offset + 2],
                    row[offset + 3],
                ]);
            }
        }
        drop(mapped);
        staging.unmap();
        pixels
    }

    /// The physical-GPU claim: the shader matches the CPU reference at
    /// sampled pixels across every shape and colour mode within the sRGB
    /// byte lattice, and the same uniforms render the same bytes twice.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_pattern_synth_matches_the_cpu_reference_for_every_shape() {
        let Some((device, queue)) = acquire_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let width = PATTERN_SYNTH_WIDTH;
        let height = PATTERN_SYNTH_HEIGHT;
        let aspect = width as f32 / height as f32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pattern fixture target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut stage = PatternSynthGpu::new(&device);
        // A dense sample grid: the comparison is statistical because shapes
        // with a discrete argmin (Cells) or steep oscillator slopes can
        // legitimately diverge by several code values at isolated pixels
        // under f32 transcendental differences, while any real law drift
        // moves the whole field.
        let mut sample_points = Vec::new();
        for gy in 0..9u32 {
            for gx in 0..16u32 {
                sample_points.push((60 + gx * 120, 60 + gy * 120));
            }
        }
        for shape in PatternShape::ALL {
            for color_mode in [
                PatternColorMode::Mono,
                PatternColorMode::RgbPhase,
                PatternColorMode::HsvSweep,
                PatternColorMode::Duotone,
                PatternColorMode::Bands,
            ] {
                let params = PatternSynthParams {
                    shape,
                    color_mode,
                    cross_mod: 0.35,
                    wavefold: 0.25,
                    comparator: 0.4,
                    warp: 0.2,
                    rotate: 0.13,
                    skew: -0.2,
                    zoom: 0.3,
                    ..PatternSynthParams::default()
                };
                let time = 3.75f32;
                let uniforms = PatternSynthGpuUniforms::from_params(&params, time);
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                stage.encode(&device, &queue, &mut encoder, &[(uniforms, &view)]);
                queue.submit(std::iter::once(encoder.finish()));
                let pixels = read_pixels(&device, &queue, &texture, width, height);
                let mut comparisons = 0usize;
                let mut agree = 0usize;
                for &(px, py) in &sample_points {
                    let uv = [
                        (px as f32 + 0.5) / width as f32,
                        (py as f32 + 0.5) / height as f32,
                    ];
                    let expected = pattern_synth_pixel(&params, uv, aspect, time);
                    let got = pixels[(py * width + px) as usize];
                    for ch in 0..3 {
                        let expected_byte = (expected[ch] * 255.0 + 0.5).floor();
                        let diff = (f32::from(got[ch]) - expected_byte).abs();
                        comparisons += 1;
                        if diff <= 4.0 {
                            agree += 1;
                        }
                    }
                    assert_eq!(got[3], 255, "the pattern page is opaque");
                }
                // 95%: BENDR's own screen hash ends in `fract(x * y)` over
                // ~4000-scale products, so one GPU/CPU ulp is amplified a
                // thousandfold before the oscillator — a few code values at
                // a few percent of pixels is the hash being itself. A wrong
                // law moves every sample by dozens and fails this bar flat.
                let fraction = agree as f64 / comparisons as f64;
                assert!(
                    fraction >= 0.95,
                    "shape {shape:?} mode {color_mode:?}: only {agree}/{comparisons} \
                     channel samples within four code values of the CPU reference"
                );
            }
        }
        // Determinism: the same uniforms render the same bytes.
        let params = PatternSynthParams::default();
        let uniforms = PatternSynthGpuUniforms::from_params(&params, 1.25);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        stage.encode(&device, &queue, &mut encoder, &[(uniforms, &view)]);
        queue.submit(std::iter::once(encoder.finish()));
        let first = read_pixels(&device, &queue, &texture, width, height);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        stage.encode(&device, &queue, &mut encoder, &[(uniforms, &view)]);
        queue.submit(std::iter::once(encoder.finish()));
        let second = read_pixels(&device, &queue, &texture, width, height);
        assert_eq!(first, second, "the pass must be deterministic per host");
    }
}
