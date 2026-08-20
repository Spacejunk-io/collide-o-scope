//! B11 monitoring-bay reduction stage.
//!
//! One 128×72 reduction of the selected probe image plus a two-slot bounded
//! readback pool, armed only while an observer is present (the native bay
//! overlay or a fresh browser watcher) and cadenced by the host on the 30 Hz
//! reference grid at 10 Hz. The machine is the B10 `video_analysis` pool at
//! the bay's dimensions, and like it, deliberately its own tiny pool rather
//! than a fourth consumer of the full-frame audience slots: those buffers
//! are fixed at full-frame size and hard-coded to copy slot 2 whole, while
//! this stage's staging is 36,864 bytes.
//!
//! The one structural difference from B10: the bay's source is the authored
//! PROBE, so the bind group is rebuilt per scheduled sample against whatever
//! retained view the probe names. At 10 Hz a bind group build is noise, and
//! it removes every epoch-tracking hazard a cached group would carry across
//! probe changes, renderer rebuilds, and canvas reallocation.
//!
//! Ledger, charged exactly and deliberately outside the full-frame texture
//! floor (the video-analysis precedent — not a full-frame surface owner):
//! one 128×72 RGBA8 target (36,864 bytes) and two 36,864-byte staging
//! buffers, allocated lazily on the first armed frame. One render pass
//! (16 bilinear taps per cell, 9,216 cells) and one copy per armed 10 Hz
//! sample; nothing at all while hidden.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub const MONITOR_BAY_WIDTH: u32 = crate::monitor_bay::MONITOR_BAY_WIDTH as u32;
pub const MONITOR_BAY_HEIGHT: u32 = crate::monitor_bay::MONITOR_BAY_HEIGHT as u32;
/// 128 texels × 4 bytes = 512 bytes per row, already a multiple of wgpu's
/// 256-byte rule.
pub const MONITOR_BAY_PADDED_ROW_BYTES: u32 = 512;
pub const MONITOR_BAY_SLOTS: usize = 2;

const SLOT_IDLE: u8 = 0;
const SLOT_IN_FLIGHT: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_FAILED: u8 = 3;

struct MonitorSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    sequence: u64,
}

pub struct MonitorBayGpu {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    slots: Vec<MonitorSlot>,
    next_sequence: u64,
}

impl MonitorBayGpu {
    pub fn new(device: &wgpu::Device, composite_format: wgpu::TextureFormat) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Monitor bay bind layout"),
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
            label: Some("Monitor bay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/monitor_bay.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Monitor bay pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Monitor bay pipeline"),
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
            label: Some("Monitor bay sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor bay target"),
            size: wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
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
        let slots = (0..MONITOR_BAY_SLOTS)
            .map(|index| MonitorSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Monitor bay readback {index}")),
                    size: u64::from(MONITOR_BAY_PADDED_ROW_BYTES) * u64::from(MONITOR_BAY_HEIGHT),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                status: Arc::new(AtomicU8::new(SLOT_IDLE)),
                sequence: 0,
            })
            .collect();
        Self {
            pipeline,
            bind_layout,
            sampler,
            target,
            target_view,
            slots,
            next_sequence: 1,
        }
    }

    /// Encode one reduction of `source_view` + copy into its own encoder and
    /// begin the map. Returns false (a clean drop, never a backlog) when
    /// both slots are still in flight.
    pub fn schedule(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_view: &wgpu::TextureView,
    ) -> bool {
        let Some(index) = self
            .slots
            .iter()
            .position(|slot| slot.status.load(Ordering::Acquire) == SLOT_IDLE)
        else {
            return false;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Monitor bay bind group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Monitor bay reduce"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Monitor bay pass"),
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
            pass.set_bind_group(0, &bind_group, &[]);
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
                    bytes_per_row: Some(MONITOR_BAY_PADDED_ROW_BYTES),
                    rows_per_image: Some(MONITOR_BAY_HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slot = &mut self.slots[index];
        slot.sequence = self.next_sequence;
        self.next_sequence += 1;
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

    /// Harvest the oldest completed sample, de-padded to 128×72×4 bytes.
    /// Failed maps recycle silently — the next cadence tick simply samples
    /// again.
    pub fn poll(&mut self) -> Option<Vec<u8>> {
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
        let grid = {
            let slot = &self.slots[index];
            let mapped = slot.buffer.slice(..).get_mapped_range();
            let mut grid = Vec::with_capacity(crate::monitor_bay::MONITOR_BAY_CELLS * 4);
            for row in 0..MONITOR_BAY_HEIGHT as usize {
                let start = row * MONITOR_BAY_PADDED_ROW_BYTES as usize;
                grid.extend_from_slice(&mapped[start..start + MONITOR_BAY_WIDTH as usize * 4]);
            }
            grid
        };
        self.slots[index].buffer.unmap();
        self.slots[index].status.store(SLOT_IDLE, Ordering::Release);
        Some(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const SOURCE_WIDTH: u32 = 512;
    const SOURCE_HEIGHT: u32 = 288;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Monitor bay test"),
            ..Default::default()
        }))
        .ok()
    }

    /// A deterministic synthetic probe image: gradients plus a hard block,
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

    /// The GPU reduction agrees with `reduce_analysis_grid` at the bay's
    /// dimensions to the B7 statistical contract, and the two-slot pool
    /// drops cleanly when saturated. A wrong law — a different tap grid,
    /// filtering in the wrong space — moves most of the 9,216 cells by far
    /// more than the tolerance and fails flat.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_monitor_bay_reduction_matches_the_cpu_reference() {
        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor bay fixture source"),
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
        let mut stage = MonitorBayGpu::new(&device, FORMAT);

        assert!(stage.schedule(&device, &queue, &source_view));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let gpu_grid = stage.poll().expect("a mapped monitor grid");
        assert_eq!(gpu_grid.len(), crate::monitor_bay::MONITOR_BAY_CELLS * 4);

        let cpu_grid = crate::modulation::reduce_analysis_grid(
            &pixels,
            SOURCE_WIDTH as usize,
            SOURCE_HEIGHT as usize,
            crate::monitor_bay::MONITOR_BAY_WIDTH,
            crate::monitor_bay::MONITOR_BAY_HEIGHT,
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

        // The two-slot pool drops cleanly when saturated — an instrument
        // wants freshness, never a backlog.
        assert!(stage.schedule(&device, &queue, &source_view));
        assert!(stage.schedule(&device, &queue, &source_view));
        assert!(
            !stage.schedule(&device, &queue, &source_view),
            "a third in-flight sample must drop, not queue"
        );
    }
}
