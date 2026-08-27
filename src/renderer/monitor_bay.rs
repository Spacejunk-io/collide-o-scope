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
//! one 128×72 RGBA8 reduction target (36,864 bytes) and two 36,864-byte
//! staging buffers, allocated lazily on the first armed frame. Deferred
//! texture probes share one additional 36,864-byte linear RGBA8 target,
//! allocated only after one of them wins an idle slot; their pipelines are
//! independently lazy. CPU RGBA uses the existing staging directly. One
//! render pass and one copy per texture-backed 10 Hz sample; nothing at all
//! while hidden.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub const MONITOR_BAY_WIDTH: u32 = crate::monitor_bay::MONITOR_BAY_WIDTH as u32;
pub const MONITOR_BAY_HEIGHT: u32 = crate::monitor_bay::MONITOR_BAY_HEIGHT as u32;
/// 128 texels × 4 bytes = 512 bytes per row, already a multiple of wgpu's
/// 256-byte rule.
pub const MONITOR_BAY_PADDED_ROW_BYTES: u32 = 512;
pub const MONITOR_BAY_SLOTS: usize = 2;
pub const MONITOR_BAY_TEXTURE_BYTES: u64 =
    MONITOR_BAY_PADDED_ROW_BYTES as u64 * MONITOR_BAY_HEIGHT as u64;

/// Retained-object and nominal-byte ledger for the lazy monitor stage.
///
/// The original image-reduction machine is present whenever `MonitorBayGpu`
/// exists. The exact-RGBA target and each deferred probe's pipeline are born
/// only after that path wins an idle FIFO slot. Per-sample bind groups are
/// deliberately transient and therefore are not retained allocations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "monitor allocation snapshots are exercised by focused GPU and ledger fixtures"
    )
)]
pub(crate) struct MonitorBayAllocationSnapshot {
    pub reduction_target_textures: u64,
    pub reduction_target_views: u64,
    pub readback_buffers: u64,
    pub reduction_shader_modules: u64,
    pub reduction_bind_group_layouts: u64,
    pub reduction_pipeline_layouts: u64,
    pub reduction_pipelines: u64,
    pub reduction_samplers: u64,
    pub reduction_target_bytes: u64,
    pub readback_buffer_bytes: u64,
    pub exact_target_textures: u64,
    pub exact_target_views: u64,
    pub exact_target_bytes: u64,
    pub rgba_view_shader_modules: u64,
    pub rgba_view_bind_group_layouts: u64,
    pub rgba_view_pipeline_layouts: u64,
    pub rgba_view_pipelines: u64,
    pub motion_shader_modules: u64,
    pub motion_bind_group_layouts: u64,
    pub motion_pipeline_layouts: u64,
    pub motion_pipelines: u64,
    pub motion_uniform_buffers: u64,
    pub motion_uniform_bytes: u64,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "monitor allocation snapshots are exercised by focused GPU and ledger fixtures"
    )
)]
const fn monitor_bay_allocation_snapshot_for(
    exact_target: bool,
    rgba_view: bool,
    motion: bool,
) -> MonitorBayAllocationSnapshot {
    MonitorBayAllocationSnapshot {
        reduction_target_textures: 1,
        reduction_target_views: 1,
        readback_buffers: MONITOR_BAY_SLOTS as u64,
        reduction_shader_modules: 1,
        reduction_bind_group_layouts: 1,
        reduction_pipeline_layouts: 1,
        reduction_pipelines: 1,
        reduction_samplers: 1,
        reduction_target_bytes: MONITOR_BAY_TEXTURE_BYTES,
        readback_buffer_bytes: MONITOR_BAY_TEXTURE_BYTES * MONITOR_BAY_SLOTS as u64,
        exact_target_textures: exact_target as u64,
        exact_target_views: exact_target as u64,
        exact_target_bytes: MONITOR_BAY_TEXTURE_BYTES * exact_target as u64,
        rgba_view_shader_modules: rgba_view as u64,
        rgba_view_bind_group_layouts: rgba_view as u64,
        rgba_view_pipeline_layouts: rgba_view as u64,
        rgba_view_pipelines: rgba_view as u64,
        motion_shader_modules: motion as u64,
        motion_bind_group_layouts: motion as u64,
        motion_pipeline_layouts: motion as u64,
        motion_pipelines: motion as u64,
        motion_uniform_buffers: motion as u64,
        motion_uniform_bytes: 16 * motion as u64,
    }
}

const SLOT_IDLE: u8 = 0;
const SLOT_IN_FLIGHT: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_FAILED: u8 = 3;

struct MonitorSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    sequence: u64,
    generation: u64,
}

struct ExactTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct RgbaViewResources {
    bind_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

struct MotionFieldResources {
    bind_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionMonitorUniforms {
    grid: [u32; 2],
    max_uv_per_second: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<MotionMonitorUniforms>() == 16);

pub struct MonitorBayGpu {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    exact_target: Option<ExactTarget>,
    rgba_view_resources: Option<RgbaViewResources>,
    motion_field_resources: Option<MotionFieldResources>,
    slots: Vec<MonitorSlot>,
    next_sequence: u64,
    generation: u64,
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
                generation: 0,
            })
            .collect();
        Self {
            pipeline,
            bind_layout,
            sampler,
            target,
            target_view,
            exact_target: None,
            rgba_view_resources: None,
            motion_field_resources: None,
            slots,
            next_sequence: 1,
            generation: 0,
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
        let Some(index) = self.idle_slot_index() else {
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
        self.begin_map(index);
        true
    }

    /// Publish one already-reduced 128×72 opaque RGBA oracle into the shared
    /// FIFO without allocating another texture or pipeline. This is the sync
    /// latch's CPU-owned path. A malformed grid and a saturated pool are both
    /// clean refusals: neither mutates sequence or mapping state.
    pub fn schedule_cpu_rgba(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
    ) -> bool {
        if rgba.len() != MONITOR_BAY_TEXTURE_BYTES as usize {
            return false;
        }
        let Some(index) = self.idle_slot_index() else {
            return false;
        };
        queue.write_buffer(&self.slots[index].buffer, 0, rgba);
        // Queue writes become visible with the next submission. An explicit
        // empty submit makes this one-shot path self-contained just like the
        // render-backed schedules below.
        queue.submit(std::iter::empty());
        self.begin_map(index);
        true
    }

    /// Read one retained RGBA texture into the bay grid through exact nearest
    /// texel selection. The dedicated linear RGBA8 target preserves the
    /// producer's bytes; routing this through the existing sRGB reduction
    /// target would gamma-encode an already diagnostic scalar image.
    pub fn schedule_rgba_view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_view: &wgpu::TextureView,
    ) -> bool {
        let Some(index) = self.idle_slot_index() else {
            return false;
        };
        self.ensure_exact_target(device);
        self.ensure_rgba_view_resources(device);
        let resources = self
            .rgba_view_resources
            .as_ref()
            .expect("RGBA-view monitor resources were materialized");
        let target = self
            .exact_target
            .as_ref()
            .expect("exact monitor target was materialized");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Monitor bay exact RGBA bind group"),
            layout: &resources.bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Monitor bay exact RGBA"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Monitor bay exact RGBA pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
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
            pass.set_pipeline(&resources.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Self::copy_target_to_slot(&mut encoder, &target.texture, &self.slots[index].buffer);
        queue.submit(std::iter::once(encoder.finish()));
        self.begin_map(index);
        true
    }

    /// Visualize the exact vector/gate parity selected by Master motion for
    /// the currently staged frame. `grid` is the admitted primitive grid or
    /// the Field Collider's `plan.output_grid`; no source is inferred here.
    pub(crate) fn schedule_motion_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vectors: &wgpu::TextureView,
        gates: &wgpu::TextureView,
        grid: [u32; 2],
    ) -> bool {
        if grid[0] == 0 || grid[1] == 0 {
            return false;
        }
        let Some(index) = self.idle_slot_index() else {
            return false;
        };
        self.ensure_exact_target(device);
        self.ensure_motion_field_resources(device);
        let resources = self
            .motion_field_resources
            .as_ref()
            .expect("motion monitor resources were materialized");
        let target = self
            .exact_target
            .as_ref()
            .expect("exact monitor target was materialized");
        queue.write_buffer(
            &resources.uniform,
            0,
            bytemuck::bytes_of(&MotionMonitorUniforms {
                grid,
                max_uv_per_second: crate::motion::MOTION_MAX_UV_PER_SECOND,
                _pad: 0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Monitor bay motion-field bind group"),
            layout: &resources.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(vectors),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(gates),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Monitor bay motion field"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Monitor bay motion-field pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
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
            pass.set_pipeline(&resources.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Self::copy_target_to_slot(&mut encoder, &target.texture, &self.slots[index].buffer);
        queue.submit(std::iter::once(encoder.finish()));
        self.begin_map(index);
        true
    }

    /// Exact retained allocation after lazy deferred-path materialization.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "monitor allocation snapshots are exercised by focused GPU and ledger fixtures"
        )
    )]
    pub(crate) const fn allocation_snapshot(&self) -> MonitorBayAllocationSnapshot {
        monitor_bay_allocation_snapshot_for(
            self.exact_target.is_some(),
            self.rgba_view_resources.is_some(),
            self.motion_field_resources.is_some(),
        )
    }

    fn idle_slot_index(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.status.load(Ordering::Acquire) == SLOT_IDLE)
    }

    fn begin_map(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        slot.sequence = self.next_sequence;
        self.next_sequence += 1;
        slot.generation = self.generation;
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
    }

    /// Invalidate every sample scheduled under the previous probe/arming
    /// epoch. Completed stale slots recycle immediately; an in-flight map
    /// keeps its slot until its callback completes, then `poll` recycles it
    /// silently. The generation stamp prevents an old completion from being
    /// mistaken for a fresh sample after re-arm.
    pub fn invalidate_samples(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("monitor sample generation exhausted");
        self.recycle_completed_stale_slots();
    }

    fn recycle_completed_stale_slots(&mut self) {
        for slot in &mut self.slots {
            let status = slot.status.load(Ordering::Acquire);
            if slot_completion_is_stale(status, slot.generation, self.generation) {
                slot.buffer.unmap();
                slot.status.store(SLOT_IDLE, Ordering::Release);
            }
        }
    }

    fn copy_target_to_slot(
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        buffer: &wgpu::Buffer,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
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
    }

    fn ensure_exact_target(&mut self, device: &wgpu::Device) {
        if self.exact_target.is_some() {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor bay exact RGBA8 target"),
            size: wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.exact_target = Some(ExactTarget { texture, view });
    }

    fn ensure_rgba_view_resources(&mut self, device: &wgpu::Device) {
        if self.rgba_view_resources.is_some() {
            return;
        }
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Monitor bay exact RGBA bind layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Monitor bay deferred RGBA shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/monitor_bay_deferred.wgsl").into(),
            ),
        });
        let pipeline = create_deferred_pipeline(
            device,
            "Monitor bay exact RGBA pipeline layout",
            "Monitor bay exact RGBA pipeline",
            &module,
            &bind_layout,
            "fs_rgba",
        );
        self.rgba_view_resources = Some(RgbaViewResources {
            bind_layout,
            pipeline,
        });
    }

    fn ensure_motion_field_resources(&mut self, device: &wgpu::Device) {
        if self.motion_field_resources.is_some() {
            return;
        }
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Monitor bay motion-field bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Monitor bay motion-field shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/monitor_bay_deferred.wgsl").into(),
            ),
        });
        let pipeline = create_deferred_pipeline(
            device,
            "Monitor bay motion-field pipeline layout",
            "Monitor bay motion-field pipeline",
            &module,
            &bind_layout,
            "fs_motion",
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monitor bay motion-field grid uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.motion_field_resources = Some(MotionFieldResources {
            bind_layout,
            pipeline,
            uniform,
        });
    }

    /// Harvest the oldest completed sample, de-padded to 128×72×4 bytes.
    /// Failed maps recycle silently — the next cadence tick simply samples
    /// again.
    pub fn poll(&mut self) -> Option<Vec<u8>> {
        self.recycle_completed_stale_slots();
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
            .filter(|(_, slot)| {
                slot.generation == self.generation
                    && slot.status.load(Ordering::Acquire) == SLOT_MAPPED
            })
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

const fn slot_completion_is_stale(status: u8, slot_generation: u64, generation: u64) -> bool {
    slot_generation != generation && (status == SLOT_MAPPED || status == SLOT_FAILED)
}

fn create_deferred_pipeline(
    device: &wgpu::Device,
    layout_label: &str,
    pipeline_label: &str,
    module: &wgpu::ShaderModule,
    bind_layout: &wgpu::BindGroupLayout,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(layout_label),
        bind_group_layouts: &[Some(bind_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_exact"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
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

    #[test]
    fn deferred_paths_have_exact_lazy_allocation_snapshots() {
        let base = MonitorBayAllocationSnapshot {
            reduction_target_textures: 1,
            reduction_target_views: 1,
            readback_buffers: 2,
            reduction_shader_modules: 1,
            reduction_bind_group_layouts: 1,
            reduction_pipeline_layouts: 1,
            reduction_pipelines: 1,
            reduction_samplers: 1,
            reduction_target_bytes: 36_864,
            readback_buffer_bytes: 73_728,
            ..MonitorBayAllocationSnapshot::default()
        };
        assert_eq!(
            monitor_bay_allocation_snapshot_for(false, false, false),
            base,
            "constructing the bay does not preallocate any deferred path"
        );
        // CPU RGBA writes directly into an idle readback slot, so using that
        // path has exactly the base allocation above.
        assert_eq!(
            monitor_bay_allocation_snapshot_for(false, false, false),
            base
        );
        assert_eq!(
            monitor_bay_allocation_snapshot_for(true, true, false),
            MonitorBayAllocationSnapshot {
                exact_target_textures: 1,
                exact_target_views: 1,
                exact_target_bytes: 36_864,
                rgba_view_shader_modules: 1,
                rgba_view_bind_group_layouts: 1,
                rgba_view_pipeline_layouts: 1,
                rgba_view_pipelines: 1,
                ..base
            }
        );
        assert_eq!(
            monitor_bay_allocation_snapshot_for(true, false, true),
            MonitorBayAllocationSnapshot {
                exact_target_textures: 1,
                exact_target_views: 1,
                exact_target_bytes: 36_864,
                motion_shader_modules: 1,
                motion_bind_group_layouts: 1,
                motion_pipeline_layouts: 1,
                motion_pipelines: 1,
                motion_uniform_buffers: 1,
                motion_uniform_bytes: 16,
                ..base
            }
        );
        assert_eq!(
            monitor_bay_allocation_snapshot_for(true, true, true),
            MonitorBayAllocationSnapshot {
                exact_target_textures: 1,
                exact_target_views: 1,
                exact_target_bytes: 36_864,
                rgba_view_shader_modules: 1,
                rgba_view_bind_group_layouts: 1,
                rgba_view_pipeline_layouts: 1,
                rgba_view_pipelines: 1,
                motion_shader_modules: 1,
                motion_bind_group_layouts: 1,
                motion_pipeline_layouts: 1,
                motion_pipelines: 1,
                motion_uniform_buffers: 1,
                motion_uniform_bytes: 16,
                ..base
            },
            "the exact target is shared rather than allocated once per path"
        );
    }

    #[test]
    fn stale_completion_cannot_cross_probe_generation_or_rearm() {
        let old_generation = 7;
        let rearmed_generation = 8;
        assert!(slot_completion_is_stale(
            SLOT_MAPPED,
            old_generation,
            rearmed_generation
        ));
        assert!(slot_completion_is_stale(
            SLOT_FAILED,
            old_generation,
            rearmed_generation
        ));
        assert!(
            !slot_completion_is_stale(SLOT_IN_FLIGHT, old_generation, rearmed_generation),
            "an old in-flight map remains busy until its callback completes"
        );
        // ABA shape: the old callback lands only after a fresh generation has
        // already been armed. Its old stamp still forces silent recycling;
        // only a completion stamped by the current arm can publish.
        assert!(slot_completion_is_stale(
            SLOT_MAPPED,
            old_generation,
            rearmed_generation
        ));
        assert!(!slot_completion_is_stale(
            SLOT_MAPPED,
            rearmed_generation,
            rearmed_generation
        ));
    }

    #[test]
    fn deferred_shader_uses_the_engine_bound_and_sanitizes_both_gate_lanes() {
        let shader = include_str!("../shaders/monitor_bay_deferred.wgsl");
        assert!(shader.contains("motion.max_uv_per_second"));
        assert!(!shader.contains("MOTION_MAX_UV_PER_SECOND"));
        assert!(shader.contains("finite_or_zero(textureLoad(motion_vectors"));
        assert!(shader.contains("finite_or_zero(textureLoad(motion_gates"));
        assert!(shader.contains("gate.x * gate.y"));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_deferred_probes_preserve_exact_bytes_and_motion_color_law() {
        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let mut stage = MonitorBayGpu::new(&device, FORMAT);
        let base = stage.allocation_snapshot();

        let mut cpu_rgba = vec![0; MONITOR_BAY_TEXTURE_BYTES as usize];
        for (index, pixel) in cpu_rgba.chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&[
                (index % 251) as u8,
                (index / 251 % 241) as u8,
                (index / 17 % 239) as u8,
                255,
            ]);
        }
        let old_rgba = vec![17; MONITOR_BAY_TEXTURE_BYTES as usize];
        assert!(stage.schedule_cpu_rgba(&device, &queue, &old_rgba));
        stage.invalidate_samples();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        assert_eq!(
            stage.poll(),
            None,
            "a mapped completion from before re-arm must recycle silently"
        );
        assert!(stage.schedule_cpu_rgba(&device, &queue, &cpu_rgba));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        assert_eq!(
            stage.poll().expect("post-rearm CPU monitor sample"),
            cpu_rgba
        );

        assert!(!stage.schedule_cpu_rgba(&device, &queue, &cpu_rgba[..cpu_rgba.len() - 1]));
        assert!(stage.schedule_cpu_rgba(&device, &queue, &cpu_rgba));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        assert_eq!(stage.poll().expect("CPU monitor sample"), cpu_rgba);
        assert_eq!(stage.allocation_snapshot(), base);

        let rgba_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor exact RGBA fixture"),
            size: wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            rgba_texture.as_image_copy(),
            &cpu_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MONITOR_BAY_PADDED_ROW_BYTES),
                rows_per_image: Some(MONITOR_BAY_HEIGHT),
            },
            wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let rgba_view = rgba_texture.create_view(&wgpu::TextureViewDescriptor::default());
        assert!(stage.schedule_rgba_view(&device, &queue, &rgba_view));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        assert_eq!(stage.poll().expect("RGBA-view monitor sample"), cpu_rgba);

        let mut vector_bytes = vec![0; MONITOR_BAY_TEXTURE_BYTES as usize];
        let gate_row_bytes = MONITOR_BAY_WIDTH * 2;
        let mut gate_bytes = vec![0; (gate_row_bytes * MONITOR_BAY_HEIGHT) as usize];
        let put = |bytes: &mut [u8], index: usize, pair: [u16; 2]| {
            let offset = index * 4;
            bytes[offset..offset + 2].copy_from_slice(&pair[0].to_le_bytes());
            bytes[offset + 2..offset + 4].copy_from_slice(&pair[1].to_le_bytes());
        };
        let put_gate = |bytes: &mut [u8], index: usize, pair: [u8; 2]| {
            bytes[index * 2..index * 2 + 2].copy_from_slice(&pair);
        };
        // Production vector/gate formats: RG16Float and RG8Unorm. Hostile
        // non-finite lanes are representable only in the vector texture; the
        // CPU oracle separately pins finite-or-zero gate sanitization before
        // RG8 encoding.
        put_gate(&mut gate_bytes, 0, [0, 255]);
        put(&mut vector_bytes, 1, [0x5400, 0x5400]);
        put_gate(&mut gate_bytes, 1, [128, 128]);
        put(&mut vector_bytes, 2, [0xd400, 0xd400]);
        put_gate(&mut gate_bytes, 2, [255, 255]);
        put(&mut vector_bytes, 3, [0x7c00, 0xfc00]);
        put_gate(&mut gate_bytes, 3, [0, 255]);
        let vectors = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor production RG16Float motion-vector fixture"),
            size: wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::renderer::motion::MOTION_VECTOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            vectors.as_image_copy(),
            &vector_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MONITOR_BAY_PADDED_ROW_BYTES),
                rows_per_image: Some(MONITOR_BAY_HEIGHT),
            },
            wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let gates = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Monitor production RG8Unorm motion-gate fixture"),
            size: wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::renderer::motion::MOTION_GATE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            gates.as_image_copy(),
            &gate_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(gate_row_bytes),
                rows_per_image: Some(MONITOR_BAY_HEIGHT),
            },
            wgpu::Extent3d {
                width: MONITOR_BAY_WIDTH,
                height: MONITOR_BAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let vector_view = vectors.create_view(&wgpu::TextureViewDescriptor::default());
        let gate_view = gates.create_view(&wgpu::TextureViewDescriptor::default());
        assert!(stage.schedule_motion_field(
            &device,
            &queue,
            &vector_view,
            &gate_view,
            [MONITOR_BAY_WIDTH, MONITOR_BAY_HEIGHT],
        ));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        let rendered = stage.poll().expect("motion monitor sample");
        assert_eq!(&rendered[0..4], &[128, 128, 0, 255]);
        assert_eq!(&rendered[4..8], &[255, 0, 64, 255]);
        assert_eq!(&rendered[8..12], &[0, 255, 255, 255]);
        assert_eq!(&rendered[12..16], &[128, 128, 0, 255]);
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
