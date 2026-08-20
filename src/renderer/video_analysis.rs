//! B10 video content-analysis reduction stage.
//!
//! One 32×18 reduction of the pre-blackout opaque audience image (composite
//! slot 2, the program-tap seam) plus a two-slot bounded readback pool, armed
//! on demand by `ModMatrix::video_analysis_armed` and cadenced by the host on
//! the 30 Hz reference grid at 10 Hz. The pool is deliberately its own tiny
//! machine rather than a fourth consumer of the full-frame audience slots:
//! those buffers are fixed at full-frame size and hard-coded to copy slot 2
//! whole, while this stage's whole point is that its staging is 4,608 bytes.
//! Harvest is strict FIFO by sequence (the recorder-readback law), and a
//! frame with no free slot simply drops the sample — a modulation source
//! wants freshness, never a backlog.
//!
//! Ledger, charged exactly and deliberately outside the full-frame texture
//! floor (the pattern-synth precedent — not a full-frame surface owner):
//! one 32×18 RGBA8 target (2,304 bytes) and two 4,608-byte staging buffers,
//! allocated lazily on the first armed frame. One render pass (16 bilinear
//! taps per cell, 576 cells) and one 2,304-byte copy per armed 10 Hz sample.
//!
//! The CPU reference is `modulation::reduce_video_analysis_grid`; the GPU
//! agreement claim is statistical (the B7 law) because filtering precision
//! differs per adapter.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub const VIDEO_ANALYSIS_WIDTH: u32 = crate::modulation::VIDEO_ANALYSIS_WIDTH as u32;
pub const VIDEO_ANALYSIS_HEIGHT: u32 = crate::modulation::VIDEO_ANALYSIS_HEIGHT as u32;
/// 32 texels × 4 bytes = 128 bytes per row, padded to wgpu's 256-byte rule.
pub const VIDEO_ANALYSIS_PADDED_ROW_BYTES: u32 = 256;
pub const VIDEO_ANALYSIS_SLOTS: usize = 2;

const SLOT_IDLE: u8 = 0;
const SLOT_IN_FLIGHT: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_FAILED: u8 = 3;

struct AnalysisSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    sequence: u64,
    /// Accepted program seconds the sample spans, carried with the pixels so
    /// the analysis law integrates the exact interval that produced them.
    dt: f32,
}

pub struct VideoAnalysisGpu {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    slots: Vec<AnalysisSlot>,
    next_sequence: u64,
}

impl VideoAnalysisGpu {
    pub fn new(
        device: &wgpu::Device,
        composite_format: wgpu::TextureFormat,
        source_view: &wgpu::TextureView,
    ) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Video analysis bind layout"),
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Video analysis shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/video_analysis.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Video analysis pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Video analysis pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_reduce"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_reduce"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: composite_format,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Video analysis sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Video analysis bind group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Video analysis target"),
            size: wgpu::Extent3d {
                width: VIDEO_ANALYSIS_WIDTH,
                height: VIDEO_ANALYSIS_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: composite_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let slots = (0..VIDEO_ANALYSIS_SLOTS)
            .map(|index| AnalysisSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Video analysis readback {index}")),
                    size: u64::from(VIDEO_ANALYSIS_PADDED_ROW_BYTES)
                        * u64::from(VIDEO_ANALYSIS_HEIGHT),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                sequence: 0,
                dt: 0.0,
            })
            .collect();
        Self {
            pipeline,
            bind_group,
            target,
            target_view,
            slots,
            next_sequence: 1,
        }
    }

    /// Encode one reduction + copy into its own encoder and begin the map.
    /// Returns false (a clean drop, never a backlog) when both slots are
    /// still in flight.
    pub fn schedule(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) -> bool {
        let Some(index) = self
            .slots
            .iter()
            .position(|slot| slot.status.load(Ordering::Acquire) == SLOT_IDLE)
        else {
            return false;
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Video analysis reduce"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Video analysis pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.slots[index].buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(VIDEO_ANALYSIS_PADDED_ROW_BYTES),
                    rows_per_image: Some(VIDEO_ANALYSIS_HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: VIDEO_ANALYSIS_WIDTH,
                height: VIDEO_ANALYSIS_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slot = &mut self.slots[index];
        slot.sequence = self.next_sequence;
        self.next_sequence += 1;
        slot.dt = dt;
        slot.status.store(SLOT_IN_FLIGHT, Ordering::Release);
        let status = slot.status.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(
                    if result.is_ok() {
                        SLOT_MAPPED
                    } else {
                        SLOT_FAILED
                    },
                    Ordering::Release,
                );
            });
        true
    }

    /// Harvest the oldest completed sample, de-padded to 32×18×4 bytes, with
    /// the program seconds it spans. Failed maps recycle silently — the next
    /// cadence tick simply samples again.
    pub fn poll(&mut self) -> Option<(Vec<u8>, f32)> {
        for slot in &mut self.slots {
            if slot.status.load(Ordering::Acquire) == SLOT_FAILED {
                slot.buffer.unmap();
                slot.status.store(SLOT_IDLE, Ordering::Release);
            }
        }
        let index = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.status.load(Ordering::Acquire) == SLOT_MAPPED)
            .min_by_key(|(_, slot)| slot.sequence)
            .map(|(index, _)| index)?;
        let (grid, dt) = {
            let slot = &self.slots[index];
            let mapped = slot.buffer.slice(..).get_mapped_range();
            let mut grid = Vec::with_capacity(crate::modulation::VIDEO_ANALYSIS_CELLS * 4);
            for row in 0..VIDEO_ANALYSIS_HEIGHT as usize {
                let start = row * VIDEO_ANALYSIS_PADDED_ROW_BYTES as usize;
                grid.extend_from_slice(&mapped[start..start + VIDEO_ANALYSIS_WIDTH as usize * 4]);
            }
            (grid, slot.dt)
        };
        self.slots[index].buffer.unmap();
        self.slots[index].status.store(SLOT_IDLE, Ordering::Release);
        Some((grid, dt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const SOURCE_WIDTH: u32 = 128;
    const SOURCE_HEIGHT: u32 = 72;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Video analysis test"),
            ..Default::default()
        }))
        .ok()
    }

    /// A deterministic synthetic program image: gradients plus a hard block,
    /// so the reduction sees both smooth and edge content.
    fn synthetic_pixels() -> Vec<u8> {
        let mut pixels = Vec::with_capacity((SOURCE_WIDTH * SOURCE_HEIGHT * 4) as usize);
        for y in 0..SOURCE_HEIGHT {
            for x in 0..SOURCE_WIDTH {
                let r = (x * 255 / (SOURCE_WIDTH - 1)) as u8;
                let g = (y * 255 / (SOURCE_HEIGHT - 1)) as u8;
                let b = if x > SOURCE_WIDTH / 2 && y > SOURCE_HEIGHT / 2 {
                    230
                } else {
                    20
                };
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }
        pixels
    }

    /// The GPU reduction agrees with `reduce_video_analysis_grid`, the shared
    /// CPU reference the export path consumes, to the B7 statistical
    /// contract: filtering precision differs per adapter, so the claim is an
    /// agreement fraction, not a per-texel bound. A wrong law — a different
    /// tap grid, filtering in the wrong space — moves most of the 2,304
    /// values by far more than the tolerance and fails flat.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_video_analysis_reduction_matches_the_cpu_reference() {
        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Video analysis fixture source"),
            size: wgpu::Extent3d {
                width: SOURCE_WIDTH,
                height: SOURCE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pixels = synthetic_pixels();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SOURCE_WIDTH * 4),
                rows_per_image: Some(SOURCE_HEIGHT),
            },
            wgpu::Extent3d {
                width: SOURCE_WIDTH,
                height: SOURCE_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let mut stage = VideoAnalysisGpu::new(&device, FORMAT, &source_view);

        assert!(stage.schedule(&device, &queue, 0.1));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let (gpu_grid, dt) = stage.poll().expect("a mapped analysis grid");
        assert_eq!(dt, 0.1, "the dt tag travels with the pixels");
        assert_eq!(gpu_grid.len(), crate::modulation::VIDEO_ANALYSIS_CELLS * 4);

        let cpu_grid = crate::modulation::reduce_video_analysis_grid(
            &pixels,
            SOURCE_WIDTH as usize,
            SOURCE_HEIGHT as usize,
        );
        let mut within = 0usize;
        let mut total_delta = 0u64;
        for (gpu, cpu) in gpu_grid.iter().zip(cpu_grid.iter()) {
            let delta = gpu.abs_diff(*cpu);
            total_delta += u64::from(delta);
            if delta <= 4 {
                within += 1;
            }
        }
        let fraction = within as f64 / gpu_grid.len() as f64;
        assert!(
            fraction >= 0.95,
            "GPU/CPU reduction agreement {fraction:.3} below the statistical contract"
        );
        assert!(
            (total_delta as f64 / gpu_grid.len() as f64) < 2.0,
            "mean reduction error must stay near the filtering noise floor"
        );

        // The two-slot pool drops cleanly when saturated — freshness, never a
        // backlog.
        assert!(stage.schedule(&device, &queue, 0.1));
        assert!(stage.schedule(&device, &queue, 0.1));
        assert!(
            !stage.schedule(&device, &queue, 0.1),
            "a third in-flight sample must drop, not queue"
        );
    }
}
