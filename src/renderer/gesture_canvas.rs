//! GPU resources for the bounded gesture vector canvas.
//!
//! Geometry, resource admission, the analytic stroke laws, the decay budget,
//! and the transactional CPU memory are owned by `crate::gesture_canvas`. This
//! adapter intentionally does not restate those laws — it creates exactly the
//! surfaces the admitted plan describes, reconciles its real byte ledger back
//! against that plan, and encodes one pass per recorded sample in recorded
//! order.
//!
//! The cell layout deliberately breaks format uniformity for a small surface,
//! exactly as `renderer/motion.rs` does: a signed `Rg16Float` vector ping-pong
//! pair and an `Rg8Unorm` coverage/hold ping-pong pair, twelve bytes a cell.
//! Both parity bind groups are built once at construction through the
//! `std::array::from_fn(|parity| …)` idiom, so a warm encode allocates nothing
//! and says so through the allocation-snapshot assertion.
//!
//! Beside the working set sits one presented donor image — the only surface
//! anything outside this module ever binds. It is written from the committed
//! parity by `fs_present`, the exact inverse of the frozen `displace_node`
//! decode, through the same layout and the same two prebuilt parity bind
//! groups. Presentation therefore adds one texture and one pipeline per canvas
//! and nothing else.

#![allow(
    dead_code,
    reason = "the frozen GPU canvas contract exposes accessors and typed errors that only the opt-in physical-GPU fixtures reach"
)]

use std::{
    fmt,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::gesture_canvas::{
    GestureCanvasError, GestureCanvasFramePlan, GestureCanvasGrid, GestureCanvasLedger,
    GestureCanvasLimits, GestureCanvasParams, GestureCanvasPlan, GestureCanvasRequest,
    GestureEtchSample, GESTURE_CANVAS_BYTES_PER_CELL, GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE,
    GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL, GESTURE_CANVAS_UNIFORM_STRIDE,
};

/// Signed displacement parity pair. Four bytes a cell each.
pub(crate) const GESTURE_CANVAS_VECTOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Coverage/hold parity pair. Two bytes a cell each.
pub(crate) const GESTURE_CANVAS_GATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

/// The single presented donor image a composition image tap binds.
///
/// It is deliberately the composition's own working format, so the route is an
/// ordinary tap reading an ordinary straight-linear premultiplied image — not a
/// second convention the rack has to learn.
pub(crate) const GESTURE_CANVAS_PRESENTED_FORMAT: wgpu::TextureFormat =
    crate::renderer::composition::COMPOSITION_WORKING_FORMAT;

/// Bytes one presented texel occupies.
pub(crate) const GESTURE_CANVAS_PRESENTED_TEXEL_BYTES: u32 = 8;

/// Bytes one `Rg16Float` texel occupies, used only to derive the readback row
/// pitch and the reconciled ledger.
pub(crate) const GESTURE_CANVAS_VECTOR_TEXEL_BYTES: u32 = 4;

/// Bytes one `Rg8Unorm` texel occupies.
pub(crate) const GESTURE_CANVAS_GATE_TEXEL_BYTES: u32 = 2;

/// The renderer's own texel accounting and the domain's frozen per-cell
/// footprint are the same twelve bytes by construction.
const _: () = assert!(
    2 * (GESTURE_CANVAS_VECTOR_TEXEL_BYTES as u64 + GESTURE_CANVAS_GATE_TEXEL_BYTES as u64)
        == GESTURE_CANVAS_BYTES_PER_CELL
);

/// One presented image, charged once. The presented class is separate from the
/// working set on both sides of the reconcile, and neither can drift without
/// the other failing to compile.
const _: () =
    assert!(GESTURE_CANVAS_PRESENTED_TEXEL_BYTES as u64 == GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GestureCanvasGpuError {
    Plan(GestureCanvasError),
    UniformSlots {
        required: usize,
        capacity: usize,
    },
    SampleCountMismatch {
        planned: u32,
        supplied: usize,
    },
    BufferSizeOverflow,
    ResourceCreation {
        context: &'static str,
        kind: &'static str,
        message: String,
    },
}

impl fmt::Display for GestureCanvasGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(formatter, "{error}"),
            Self::UniformSlots {
                required,
                capacity,
            } => write!(
                formatter,
                "gesture canvas update needs {required} uniform slots; capacity is {capacity}"
            ),
            Self::SampleCountMismatch { planned, supplied } => write!(
                formatter,
                "gesture canvas frame plan applied {planned} samples but the renderer received {supplied}"
            ),
            Self::BufferSizeOverflow => {
                formatter.write_str("gesture canvas uniform byte arithmetic overflowed")
            }
            Self::ResourceCreation {
                context,
                kind,
                message,
            } => write!(formatter, "{context} failed with a {kind} error: {message}"),
        }
    }
}

impl std::error::Error for GestureCanvasGpuError {}

impl From<GestureCanvasError> for GestureCanvasGpuError {
    fn from(error: GestureCanvasError) -> Self {
        Self::Plan(error)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GestureCanvasAllocationSnapshot {
    pub textures: u64,
    pub buffers: u64,
    pub bind_groups: u64,
    pub pipelines: u64,
}

#[derive(Default)]
struct GestureCanvasAllocationCounters {
    textures: AtomicU64,
    buffers: AtomicU64,
    bind_groups: AtomicU64,
    pipelines: AtomicU64,
}

impl GestureCanvasAllocationCounters {
    fn snapshot(&self) -> GestureCanvasAllocationSnapshot {
        GestureCanvasAllocationSnapshot {
            textures: self.textures.load(Ordering::Relaxed),
            buffers: self.buffers.load(Ordering::Relaxed),
            bind_groups: self.bind_groups.load(Ordering::Relaxed),
            pipelines: self.pipelines.load(Ordering::Relaxed),
        }
    }
}

/// What one encoded update actually did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GestureCanvasEncodeReport {
    pub executed_passes: u32,
    pub decay_ticks: u32,
    pub applied_samples: u32,
    /// Parity holding the committed field after this update.
    pub committed_parity: u8,
}

/// Packed uniform block for one etch pass. Each pass owns one 256-byte slot,
/// so the frozen stride addresses the same slot on every adapter.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GestureEtchGpuUniforms {
    grid_size: [u32; 2],
    decay_ticks: f32,
    retention: f32,
    /// x, y, pressure, radius.
    sample: [f32; 4],
    /// axis x, axis y, strength, active.
    axis: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GestureEtchGpuUniforms>() == 48);
const _: () =
    assert!(std::mem::size_of::<GestureEtchGpuUniforms>() as u64 <= GESTURE_CANVAS_UNIFORM_STRIDE);

impl GestureEtchGpuUniforms {
    fn decay_only(grid: GestureCanvasGrid, params: GestureCanvasParams, decay_ticks: u32) -> Self {
        Self {
            grid_size: [grid.width, grid.height],
            decay_ticks: decay_ticks as f32,
            retention: params.retention,
            sample: [0.0; 4],
            axis: [0.0; 4],
        }
    }

    fn for_sample(
        grid: GestureCanvasGrid,
        params: GestureCanvasParams,
        decay_ticks: u32,
        sample: GestureEtchSample,
    ) -> Self {
        Self {
            grid_size: [grid.width, grid.height],
            decay_ticks: decay_ticks as f32,
            retention: params.retention,
            sample: [
                sample.position[0],
                sample.position[1],
                sample.pressure,
                params.radius,
            ],
            axis: [sample.axis[0], sample.axis[1], params.strength, 1.0],
        }
    }
}

struct GestureCanvasTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl GestureCanvasTexture {
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        grid: GestureCanvasGrid,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: grid.width,
                height: grid.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

/// One prepared gesture canvas.
pub(crate) struct GestureCanvasGpu {
    grid: GestureCanvasGrid,
    vectors: [GestureCanvasTexture; 2],
    gates: [GestureCanvasTexture; 2],
    /// The one routable donor image. Written from the committed parity at the
    /// end of every update that changed the field, and cleared by every reset
    /// cause, so nothing downstream can ever bind a stale etched field.
    presented: GestureCanvasTexture,
    /// Prebuilt read bind group per parity. Nothing else is created after
    /// construction, which is what the warm-encode assertion proves. The
    /// presentation pass reuses these same two groups — it is the same
    /// layout with the uniform simply unread — so publishing the donor adds no
    /// bind group and no bind group layout.
    etch_groups: [wgpu::BindGroup; 2],
    pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_slots: usize,
    parity: usize,
    allocations: GestureCanvasAllocationCounters,
}

impl GestureCanvasGpu {
    /// Create one canvas's surfaces. The grid has already cleared the domain's
    /// edge, cell-count, byte, and device gates inside `GestureCanvasPlan`; the
    /// only checks here are the ones about *this* canvas's uniform buffer.
    fn new(device: &wgpu::Device, grid: GestureCanvasGrid) -> Result<Self, GestureCanvasGpuError> {
        let uniform_size = std::mem::size_of::<GestureEtchGpuUniforms>() as u64;
        let uniform_slots = GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE;
        let buffer_size = GESTURE_CANVAS_UNIFORM_STRIDE
            .checked_mul(
                u64::try_from(uniform_slots)
                    .map_err(|_| GestureCanvasGpuError::BufferSizeOverflow)?,
            )
            .ok_or(GestureCanvasGpuError::BufferSizeOverflow)?;

        let allocations = GestureCanvasAllocationCounters::default();
        let resources = create_checked(device, "Gesture canvas initialization", || {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gesture canvas etch BGL"),
                entries: &[
                    sampled_texture_entry(0),
                    sampled_texture_entry(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(uniform_size),
                        },
                        count: None,
                    },
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Gesture canvas etch shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/gesture_etch.wgsl").into(),
                ),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gesture canvas etch pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Gesture canvas etch pipeline"),
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
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: GESTURE_CANVAS_VECTOR_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: GESTURE_CANVAS_GATE_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            // Same module, same layout, same prebuilt parity groups: only the
            // fragment entry point and the single colour target differ.
            let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Gesture canvas present pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_present"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: GESTURE_CANVAS_PRESENTED_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
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
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gesture canvas warmed etch uniforms"),
                size: buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let vectors: [GestureCanvasTexture; 2] = std::array::from_fn(|parity| {
                GestureCanvasTexture::new(
                    device,
                    if parity == 0 {
                        "Gesture canvas ping vector RG16Float"
                    } else {
                        "Gesture canvas pong vector RG16Float"
                    },
                    grid,
                    GESTURE_CANVAS_VECTOR_FORMAT,
                )
            });
            let gates: [GestureCanvasTexture; 2] = std::array::from_fn(|parity| {
                GestureCanvasTexture::new(
                    device,
                    if parity == 0 {
                        "Gesture canvas ping coverage/hold RG8Unorm"
                    } else {
                        "Gesture canvas pong coverage/hold RG8Unorm"
                    },
                    grid,
                    GESTURE_CANVAS_GATE_FORMAT,
                )
            });
            let presented = GestureCanvasTexture::new(
                device,
                "Gesture canvas presented donor RGBA16Float",
                grid,
                GESTURE_CANVAS_PRESENTED_FORMAT,
            );
            // Both parities are prepared here, once. A warm encode selects one
            // of these two rather than creating a third.
            let etch_groups: [wgpu::BindGroup; 2] = std::array::from_fn(|parity| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Gesture canvas prepared etch parity BG"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&vectors[parity].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&gates[parity].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform_buffer,
                                offset: 0,
                                size: NonZeroU64::new(uniform_size),
                            }),
                        },
                    ],
                })
            });
            (
                pipeline,
                present_pipeline,
                uniform_buffer,
                vectors,
                gates,
                presented,
                etch_groups,
            )
        })?;
        let (pipeline, present_pipeline, uniform_buffer, vectors, gates, presented, etch_groups) =
            resources;
        allocations.textures.fetch_add(5, Ordering::Relaxed);
        allocations.buffers.fetch_add(1, Ordering::Relaxed);
        allocations.bind_groups.fetch_add(2, Ordering::Relaxed);
        allocations.pipelines.fetch_add(2, Ordering::Relaxed);

        Ok(Self {
            grid,
            vectors,
            gates,
            presented,
            etch_groups,
            pipeline,
            present_pipeline,
            uniform_buffer,
            uniform_slots,
            parity: 0,
            allocations,
        })
    }

    pub(crate) const fn grid(&self) -> GestureCanvasGrid {
        self.grid
    }

    pub(crate) const fn committed_parity(&self) -> u8 {
        self.parity as u8
    }

    /// The committed field's views, for whatever samples the canvas.
    pub(crate) fn vector_view(&self) -> &wgpu::TextureView {
        &self.vectors[self.parity].view
    }

    pub(crate) fn gate_view(&self) -> &wgpu::TextureView {
        &self.gates[self.parity].view
    }

    pub(crate) fn vector_texture(&self) -> &wgpu::Texture {
        &self.vectors[self.parity].texture
    }

    pub(crate) fn gate_texture(&self) -> &wgpu::Texture {
        &self.gates[self.parity].texture
    }

    /// The one routable donor view a composition image tap binds.
    ///
    /// It is stable for this canvas's whole lifetime, which is what lets the
    /// executor build its tap bind groups once at prepare time and keep them
    /// across frames. A canvas rebuild produces a new view, and the executor's
    /// gesture epoch is what forces it to re-prepare.
    pub(crate) fn presented_view(&self) -> &wgpu::TextureView {
        &self.presented.view
    }

    pub(crate) fn presented_texture(&self) -> &wgpu::Texture {
        &self.presented.texture
    }

    pub(crate) fn allocation_snapshot(&self) -> GestureCanvasAllocationSnapshot {
        self.allocations.snapshot()
    }

    /// Bytes this canvas's four working surfaces actually occupy.
    fn bytes(&self) -> Result<u64, GestureCanvasError> {
        self.grid.bytes()
    }

    /// Bytes this canvas's presented donor image actually occupies.
    fn presented_bytes(&self) -> Result<u64, GestureCanvasError> {
        self.grid.presented_bytes()
    }

    /// Clear both parities and the presented donor. This is what a hard reset
    /// cause encodes; it is deliberately a GPU clear rather than an upload so
    /// no host staging buffer is needed for a reset.
    ///
    /// The presented image is cleared with the parities rather than being left
    /// to the next update, because a reset that only rewound the working set
    /// would leave the previous etched field bound and readable until something
    /// else happened to etch. A cleared presented image is `(0,0,0,0)`, which
    /// the frozen decode reads as exactly zero displacement.
    pub(crate) fn encode_clear(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for parity in 0..2 {
            clear_target(encoder, &self.vectors[parity].view);
            clear_target(encoder, &self.gates[parity].view);
        }
        clear_target(encoder, &self.presented.view);
        self.parity = 0;
    }

    /// Publish the committed parity as the routable donor image.
    ///
    /// One pass, no allocation, and the same prebuilt parity bind group the
    /// etch passes use. Presenting twice from the same committed parity is
    /// idempotent, so a caller that presents defensively cannot drift.
    pub(crate) fn encode_present(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gesture canvas present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.presented.view,
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
        pass.set_pipeline(&self.present_pipeline);
        pass.set_bind_group(0, &self.etch_groups[self.parity], &[0]);
        pass.draw(0..3, 0..1);
    }

    /// Encode one staged frame: the decay, then every applied sample in
    /// recorded order, one pass each.
    ///
    /// A held or exact-bypass frame encodes nothing at all and leaves the
    /// committed parity byte-identical — the same delegation law the rack's
    /// exact-bypass nodes obey.
    pub(crate) fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: GestureCanvasParams,
        frame: GestureCanvasFramePlan,
        samples: &[GestureEtchSample],
    ) -> Result<GestureCanvasEncodeReport, GestureCanvasGpuError> {
        if usize::try_from(frame.applied_samples) != Ok(samples.len()) {
            return Err(GestureCanvasGpuError::SampleCountMismatch {
                planned: frame.applied_samples,
                supplied: samples.len(),
            });
        }
        if frame.held || frame.is_exact_bypass() {
            return Ok(GestureCanvasEncodeReport {
                executed_passes: 0,
                decay_ticks: 0,
                applied_samples: 0,
                committed_parity: self.parity as u8,
            });
        }
        // A decay-only frame still needs one pass to carry the decay; a frame
        // with samples folds the decay into the first of them.
        let required = samples.len().max(1);
        if required > self.uniform_slots {
            return Err(GestureCanvasGpuError::UniformSlots {
                required,
                capacity: self.uniform_slots,
            });
        }

        let params = params.sanitized();
        let allocations_before = self.allocation_snapshot();
        for slot in 0..required {
            let decay_ticks = if slot == 0 { frame.decay_ticks } else { 0 };
            let uniforms = match samples.get(slot) {
                Some(sample) => {
                    GestureEtchGpuUniforms::for_sample(self.grid, params, decay_ticks, *sample)
                }
                None => GestureEtchGpuUniforms::decay_only(self.grid, params, decay_ticks),
            };
            let offset = GESTURE_CANVAS_UNIFORM_STRIDE
                .checked_mul(slot as u64)
                .ok_or(GestureCanvasGpuError::BufferSizeOverflow)?;
            queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));
            let write = 1 - self.parity;
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gesture canvas etch"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.vectors[write].view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.gates[write].view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(
                    0,
                    &self.etch_groups[self.parity],
                    &[u32::try_from(offset)
                        .map_err(|_| GestureCanvasGpuError::BufferSizeOverflow)?],
                );
                pass.draw(0..3, 0..1);
            }
            self.parity = write;
        }
        // The committed parity has moved, so republish the donor inside the
        // same encoder. A held or exact-bypass frame returned above without
        // touching either, which is why the presented image needs no republish
        // there: it already carries the committed field.
        self.encode_present(encoder);
        debug_assert_eq!(allocations_before, self.allocation_snapshot());

        Ok(GestureCanvasEncodeReport {
            executed_passes: u32::try_from(required).unwrap_or(u32::MAX),
            decay_ticks: frame.decay_ticks,
            applied_samples: frame.applied_samples,
            committed_parity: self.parity as u8,
        })
    }
}

/// Every prepared gesture canvas in one composition, plus the reconciled
/// ledger that proves the resources match the plan.
pub(crate) struct GestureCanvasResources {
    plan: GestureCanvasPlan,
    canvases: Vec<GestureCanvasGpu>,
}

impl GestureCanvasResources {
    /// Admit and then create. Preflight runs first and in full: every limit is
    /// checked before a single texture exists, and the resulting ledger is
    /// reconciled back against the plan so a renderer that quietly changed a
    /// format fails closed instead of shipping a canvas the CPU reference no
    /// longer describes.
    pub(crate) fn prepare(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        requests: &[GestureCanvasRequest],
        limits: GestureCanvasLimits,
    ) -> Result<Self, GestureCanvasGpuError> {
        let plan = GestureCanvasPlan::preflight(requests, limits)?;
        let mut canvases = Vec::with_capacity(requests.len());
        for request in requests {
            canvases.push(GestureCanvasGpu::new(device, request.grid)?);
        }
        let mut resources = Self { plan, canvases };
        plan.reconcile(resources.ledger()?)?;
        // Every surface starts at a defined, decodes-to-exactly-zero state
        // before anything can bind it, exactly as the composition executor
        // initializes its own retained surfaces before its first encode.
        let mut initialize = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gesture canvas initialization"),
        });
        resources.encode_clear(&mut initialize);
        queue.submit(std::iter::once(initialize.finish()));
        Ok(resources)
    }

    /// Encode every reset cause's device half: both working parities and the
    /// presented donor return to transparent black on every canvas.
    ///
    /// The domain owns *which* causes reset — this is the single seam through
    /// which any of them reaches the GPU, so a cause that resets the CPU field
    /// cannot quietly leave a stale etched field bound.
    pub(crate) fn encode_clear(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for canvas in &mut self.canvases {
            canvas.encode_clear(encoder);
        }
    }

    /// Encode the open CPU transaction of `state` onto canvas `index`.
    ///
    /// This is the single seam live rendering and offline export both call, so
    /// the two cannot drift into two encoders with two orderings. Everything
    /// that decides *what* the frame is — the reference-tick address, the decay
    /// budget, the modulated parameters, the ordered samples — was already
    /// decided by the CPU reference when the frame was staged; this only
    /// replays that decision onto the device.
    pub(crate) fn encode_staged_frame(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        index: usize,
        state: &crate::gesture_canvas::GestureCanvasState,
    ) -> Result<GestureCanvasEncodeReport, GestureCanvasGpuError> {
        let Some(canvas) = self.canvases.get_mut(index) else {
            return Ok(GestureCanvasEncodeReport::default());
        };
        canvas.encode(
            queue,
            encoder,
            state.staged_params(),
            state.staged_plan(),
            state.staged_samples(),
        )
    }

    /// The byte ledger the created resources actually occupy.
    pub(crate) fn ledger(&self) -> Result<GestureCanvasLedger, GestureCanvasError> {
        let mut total_bytes = 0_u64;
        let mut presented_bytes = 0_u64;
        for canvas in &self.canvases {
            total_bytes = total_bytes
                .checked_add(canvas.bytes()?)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            presented_bytes = presented_bytes
                .checked_add(canvas.presented_bytes()?)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
        }
        Ok(GestureCanvasLedger {
            canvases: u32::try_from(self.canvases.len())
                .map_err(|_| GestureCanvasError::ArithmeticOverflow)?,
            bytes_per_cell: 2
                * (u64::from(GESTURE_CANVAS_VECTOR_TEXEL_BYTES)
                    + u64::from(GESTURE_CANVAS_GATE_TEXEL_BYTES)),
            uniform_stride: GESTURE_CANVAS_UNIFORM_STRIDE,
            total_bytes,
            presented_bytes_per_cell: u64::from(GESTURE_CANVAS_PRESENTED_TEXEL_BYTES),
            presented_bytes,
        })
    }

    pub(crate) const fn plan(&self) -> GestureCanvasPlan {
        self.plan
    }

    pub(crate) fn canvas(&self, index: usize) -> Option<&GestureCanvasGpu> {
        self.canvases.get(index)
    }

    pub(crate) fn canvas_mut(&mut self, index: usize) -> Option<&mut GestureCanvasGpu> {
        self.canvases.get_mut(index)
    }

    pub(crate) fn canvas_count(&self) -> usize {
        self.canvases.len()
    }

    /// The routable donor view of the primary canvas — the one the master scope
    /// owns and the one `SavedImageSource::GestureCanvas` names. A composition
    /// with no admitted canvas has none, and the route then resolves to the
    /// planner's named `GestureCanvasUnavailable` transparency.
    pub(crate) fn presented_view(&self) -> Option<&wgpu::TextureView> {
        self.canvases.first().map(GestureCanvasGpu::presented_view)
    }

    pub(crate) fn allocation_snapshot(&self) -> GestureCanvasAllocationSnapshot {
        let mut total = GestureCanvasAllocationSnapshot::default();
        for canvas in &self.canvases {
            let snapshot = canvas.allocation_snapshot();
            total.textures += snapshot.textures;
            total.buffers += snapshot.buffers;
            total.bind_groups += snapshot.bind_groups;
            total.pipelines += snapshot.pipelines;
        }
        total
    }
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn clear_target(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
    let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Gesture canvas clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
}

fn create_checked<T>(
    device: &wgpu::Device,
    context: &'static str,
    create: impl FnOnce() -> T,
) -> Result<T, GestureCanvasGpuError> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let value = create();
    let errors = [
        ("out of memory", pollster::block_on(out_of_memory.pop())),
        ("internal/backend", pollster::block_on(internal.pop())),
        ("validation", pollster::block_on(validation.pop())),
    ];
    if let Some((kind, error)) = errors
        .into_iter()
        .find_map(|(kind, error)| error.map(|error| (kind, error)))
    {
        return Err(GestureCanvasGpuError::ResourceCreation {
            context,
            kind,
            message: error.to_string(),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gesture::{
        GestureEvent, GestureEventRecorder, GestureMode, GesturePhase, GestureTrack,
    };
    use crate::gesture_canvas::{
        GestureCanvasField, GestureCanvasFrameInput, GestureCanvasState,
        GESTURE_CANVAS_AGGREGATE_MAX_BYTES, GESTURE_CANVAS_MAX_BYTES, GESTURE_CANVAS_MAX_EDGE,
    };
    use crate::render_export::export_temporal_reference_tick;

    /// Binary16 comparison bound, matching the established Collision Rack
    /// fixtures: the vector parity is `Rg16Float`, so a per-channel `0.01` is
    /// the accepted tolerance for a GPU-versus-CPU-reference claim.
    const VECTOR_TOLERANCE: f32 = 0.01;

    /// The gate parity is `Rg8Unorm`, so each pass rounds coverage and hold to
    /// a 1/255 lattice. A fixture that encodes N passes therefore accepts N
    /// steps of that lattice rather than the binary16 bound.
    fn gate_tolerance(passes: u32) -> f32 {
        (passes as f32 + 1.0) / 255.0
    }

    fn grid(width: u32, height: u32) -> GestureCanvasGrid {
        GestureCanvasGrid::new(width, height).expect("admitted grid")
    }

    fn limits() -> GestureCanvasLimits {
        GestureCanvasLimits::device(GESTURE_CANVAS_MAX_EDGE, 256)
    }

    fn event(
        stroke: u8,
        mode: GestureMode,
        position: [f32; 2],
        pressure: f32,
        direction: [f32; 2],
    ) -> GestureEvent {
        GestureEvent::quantized(
            stroke,
            GesturePhase::Move,
            mode,
            position,
            pressure,
            direction,
        )
    }

    fn authored_params(radius: f32, strength: f32, retention: f32) -> GestureCanvasParams {
        GestureCanvasParams {
            radius,
            strength,
            retention,
        }
    }

    /// Hand-rolled binary16 decode, matching `renderer/rack.rs`'s fixture
    /// helper. The rack copy lives inside its own private test module, so this
    /// is a deliberate second copy rather than an unreachable import.
    fn f16_to_f32(value: u16) -> f32 {
        let sign = u32::from(value & 0x8000) << 16;
        let exponent = (value >> 10) & 0x1f;
        let fraction = value & 0x03ff;
        let bits = if exponent == 0 {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = u32::from(fraction);
                let mut exponent = -14_i32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                fraction &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | (u32::from(fraction) << 13)
        } else {
            sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13)
        };
        f32::from_bits(bits)
    }

    struct GpuHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl GpuHarness {
        /// The production device request, byte for byte: `Features::empty()`
        /// plus `Limits::default()`. Requesting looser limits would void the
        /// portability claim, because wgpu validates against the requested
        /// limits rather than the adapter's raw capability.
        fn new() -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("GPU adapter for gesture canvas test");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Gesture canvas test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("GPU device for gesture canvas test");
            // A GPU claim must name the adapter it was made on, exactly as the
            // StageMap physical fixtures do.
            let info = adapter.get_info();
            eprintln!(
                "Gesture canvas GPU receipt: name={}, backend={:?}, device_type={:?}, driver={}, driver_info={}",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            );
            Self { device, queue }
        }

        fn read_texture(
            &self,
            texture: &wgpu::Texture,
            grid: GestureCanvasGrid,
            texel_bytes: u32,
        ) -> Vec<u8> {
            let unpadded_row = grid.width * texel_bytes;
            let padded_row = (unpadded_row + 255) & !255;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gesture canvas test readback"),
                size: u64::from(padded_row) * u64::from(grid.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture canvas test readback encoder"),
                });
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(grid.height),
                    },
                },
                wgpu::Extent3d {
                    width: grid.width,
                    height: grid.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (send, receive) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU wait");
            receive.recv().expect("map callback").expect("map result");
            let mapped = slice.get_mapped_range();
            let mut compact = Vec::with_capacity(unpadded_row as usize * grid.height as usize);
            for row in 0..grid.height as usize {
                let start = row * padded_row as usize;
                compact.extend_from_slice(&mapped[start..start + unpadded_row as usize]);
            }
            drop(mapped);
            staging.unmap();
            compact
        }

        /// Read the presented donor image back as straight `f32` RGBA texels.
        ///
        /// `Rgba16Float` decodes through the same hand-rolled binary16 helper
        /// the vector parity uses, so the presented claim is checked with the
        /// same arithmetic and the same declared tolerance.
        fn read_presented(&self, canvas: &GestureCanvasGpu) -> Vec<[f32; 4]> {
            let grid = canvas.grid();
            let bytes = self.read_texture(
                canvas.presented_texture(),
                grid,
                GESTURE_CANVAS_PRESENTED_TEXEL_BYTES,
            );
            bytes
                .chunks_exact(8)
                .map(|texel| {
                    [
                        f16_to_f32(u16::from_le_bytes([texel[0], texel[1]])),
                        f16_to_f32(u16::from_le_bytes([texel[2], texel[3]])),
                        f16_to_f32(u16::from_le_bytes([texel[4], texel[5]])),
                        f16_to_f32(u16::from_le_bytes([texel[6], texel[7]])),
                    ]
                })
                .collect()
        }

        /// Read the committed parity back as decoded cells.
        fn read_field(&self, canvas: &GestureCanvasGpu) -> Vec<(f32, f32, f32, f32)> {
            let grid = canvas.grid();
            let vectors = self.read_texture(
                canvas.vector_texture(),
                grid,
                GESTURE_CANVAS_VECTOR_TEXEL_BYTES,
            );
            let gates =
                self.read_texture(canvas.gate_texture(), grid, GESTURE_CANVAS_GATE_TEXEL_BYTES);
            vectors
                .chunks_exact(4)
                .zip(gates.chunks_exact(2))
                .map(|(vector, gate)| {
                    (
                        f16_to_f32(u16::from_le_bytes([vector[0], vector[1]])),
                        f16_to_f32(u16::from_le_bytes([vector[2], vector[3]])),
                        f32::from(gate[0]) / 255.0,
                        f32::from(gate[1]) / 255.0,
                    )
                })
                .collect()
        }

        /// Drive one staged frame through both the CPU reference and the GPU.
        fn advance(
            &self,
            state: &mut GestureCanvasState,
            canvas: &mut GestureCanvasGpu,
            input: GestureCanvasFrameInput<'_>,
        ) -> GestureCanvasEncodeReport {
            let plan = state.stage_frame(input).expect("staged frame");
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture canvas test encoder"),
                });
            let report = canvas
                .encode(
                    &self.queue,
                    &mut encoder,
                    state.params(),
                    plan,
                    state.staged_samples(),
                )
                .expect("encoded frame");
            self.queue.submit(std::iter::once(encoder.finish()));
            state.commit_staged();
            report
        }
    }

    fn assert_matches_reference(
        readback: &[(f32, f32, f32, f32)],
        field: &GestureCanvasField,
        passes: u32,
        label: &str,
    ) {
        let gate_tolerance = gate_tolerance(passes);
        assert_eq!(readback.len(), field.cells().len(), "{label}: cell count");
        for (index, (gpu, cell)) in readback.iter().zip(field.cells()).enumerate() {
            assert!(
                (gpu.0 - cell.vector[0]).abs() <= VECTOR_TOLERANCE,
                "{label}: cell {index} vector.x {} vs reference {}",
                gpu.0,
                cell.vector[0]
            );
            assert!(
                (gpu.1 - cell.vector[1]).abs() <= VECTOR_TOLERANCE,
                "{label}: cell {index} vector.y {} vs reference {}",
                gpu.1,
                cell.vector[1]
            );
            assert!(
                (gpu.2 - cell.coverage).abs() <= gate_tolerance,
                "{label}: cell {index} coverage {} vs reference {}",
                gpu.2,
                cell.coverage
            );
            assert!(
                (gpu.3 - cell.hold).abs() <= gate_tolerance,
                "{label}: cell {index} hold {} vs reference {}",
                gpu.3,
                cell.hold
            );
        }
    }

    #[test]
    fn gesture_canvas_resources_reconcile_their_ledger_against_the_admitted_plan() {
        // Pure CPU: the ledger arithmetic the GPU path reconciles is provable
        // without an adapter, so the reconcile law is not only tested behind an
        // opt-in fixture.
        let plan = GestureCanvasPlan::preflight(
            &[
                GestureCanvasRequest::new(grid(16, 16)),
                GestureCanvasRequest::new(grid(8, 4)),
            ],
            limits(),
        )
        .expect("admitted plan");
        let ledger = GestureCanvasLedger {
            canvases: 2,
            bytes_per_cell: 2
                * (u64::from(GESTURE_CANVAS_VECTOR_TEXEL_BYTES)
                    + u64::from(GESTURE_CANVAS_GATE_TEXEL_BYTES)),
            uniform_stride: GESTURE_CANVAS_UNIFORM_STRIDE,
            total_bytes: (16 * 16 + 8 * 4) * GESTURE_CANVAS_BYTES_PER_CELL,
            presented_bytes_per_cell: u64::from(GESTURE_CANVAS_PRESENTED_TEXEL_BYTES),
            presented_bytes: (16 * 16 + 8 * 4) * GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
        };
        assert_eq!(ledger.bytes_per_cell, GESTURE_CANVAS_BYTES_PER_CELL);
        assert_eq!(
            ledger.presented_bytes_per_cell,
            GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL
        );
        assert_eq!(plan.reconcile(ledger), Ok(()));
        assert_eq!(
            GestureCanvasGpuError::from(GestureCanvasError::ArithmeticOverflow),
            GestureCanvasGpuError::Plan(GestureCanvasError::ArithmeticOverflow)
        );
    }

    #[test]
    fn the_etch_uniform_block_fits_one_frozen_two_hundred_and_fifty_six_byte_slot() {
        assert_eq!(std::mem::size_of::<GestureEtchGpuUniforms>(), 48);
        assert!(
            std::mem::size_of::<GestureEtchGpuUniforms>() as u64 <= GESTURE_CANVAS_UNIFORM_STRIDE
        );
        assert_eq!(GESTURE_CANVAS_VECTOR_FORMAT, wgpu::TextureFormat::Rg16Float);
        assert_eq!(GESTURE_CANVAS_GATE_FORMAT, wgpu::TextureFormat::Rg8Unorm);
        assert_eq!(GESTURE_CANVAS_VECTOR_TEXEL_BYTES, 4);
        assert_eq!(GESTURE_CANVAS_GATE_TEXEL_BYTES, 2);

        // An inert or decay-only pass carries an inactive axis, so the shader
        // skips the etch branch instead of running it with an invented one.
        let decay =
            GestureEtchGpuUniforms::decay_only(grid(4, 4), GestureCanvasParams::default(), 12);
        assert_eq!(decay.axis[3], 0.0);
        assert_eq!(decay.decay_ticks, 12.0);
        let etch = GestureEtchGpuUniforms::for_sample(
            grid(4, 4),
            authored_params(0.4, 0.5, 1.0),
            0,
            GestureEtchSample {
                position: [0.25, 0.75],
                pressure: 0.5,
                axis: [0.0, 1.0],
            },
        );
        assert_eq!(etch.axis, [0.0, 1.0, 0.5, 1.0]);
        assert_eq!(etch.sample, [0.25, 0.75, 0.5, 0.4]);
        assert_eq!(etch.decay_ticks, 0.0);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_gesture_canvas_push_and_curl_match_the_cpu_reference_and_a_held_frame_is_byte_identical()
    {
        let gpu = GpuHarness::new();
        let canvas_grid = grid(8, 8);
        let params = authored_params(0.35, 0.5, 1.0);
        let mut resources = GestureCanvasResources::prepare(
            &gpu.device,
            &gpu.queue,
            &[GestureCanvasRequest::new(canvas_grid)],
            limits(),
        )
        .expect("prepared canvas");
        // Four working parities plus one presented donor, and two pipelines
        // sharing one module and one layout. The presented image adds no bind
        // group at all: presentation reuses the two prebuilt parity groups.
        assert_eq!(
            resources.allocation_snapshot(),
            GestureCanvasAllocationSnapshot {
                textures: 5,
                buffers: 1,
                bind_groups: 2,
                pipelines: 2,
            }
        );
        let mut state = GestureCanvasState::new(canvas_grid, params).expect("canvas state");
        let canvas = resources.canvas_mut(0).expect("canvas");

        // Both stroke laws in one update: a Push along +x and a Curl whose
        // displacement must be perpendicular to its own stroke.
        let events = [
            event(0, GestureMode::Push, [0.3, 0.3], 1.0, [1.0, 0.0]),
            event(1, GestureMode::Curl, [0.7, 0.7], 0.75, [1.0, 0.0]),
        ];
        let allocations_before = canvas.allocation_snapshot();
        let report = gpu.advance(
            &mut state,
            canvas,
            GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &events,
                evaluated_params: None,
            },
        );
        assert_eq!(report.executed_passes, 2);
        assert_eq!(report.applied_samples, 2);
        assert_eq!(report.decay_ticks, 0);
        assert_eq!(report.committed_parity, 0);
        assert_eq!(
            allocations_before,
            canvas.allocation_snapshot(),
            "a warm encode allocated"
        );
        let etched = gpu.read_field(canvas);
        assert_matches_reference(&etched, state.field(), 2, "push and curl");

        // The Curl target cell displaces on +y for a +x stroke, and the Push
        // target cell on +x, so the two laws are visibly distinct on the GPU as
        // well as in the reference.
        let push_cell = canvas_grid.index(2, 2).expect("push cell");
        let curl_cell = canvas_grid.index(5, 5).expect("curl cell");
        assert!(etched[push_cell].0 > 0.3 && etched[push_cell].1.abs() < 0.05);
        assert!(etched[curl_cell].1 > 0.2 && etched[curl_cell].0.abs() < 0.05);

        // A Program-Freeze frame encodes nothing and leaves the canvas byte
        // identical, and a decay-only frame encodes exactly one pass.
        let vector_bytes = gpu.read_texture(
            canvas.vector_texture(),
            canvas_grid,
            GESTURE_CANVAS_VECTOR_TEXEL_BYTES,
        );
        let gate_bytes = gpu.read_texture(
            canvas.gate_texture(),
            canvas_grid,
            GESTURE_CANVAS_GATE_TEXEL_BYTES,
        );
        let held = gpu.advance(
            &mut state,
            canvas,
            GestureCanvasFrameInput {
                reference_tick: 300,
                program_advances: false,
                events: &events,
                evaluated_params: None,
            },
        );
        assert_eq!(held.executed_passes, 0);
        assert_eq!(held.committed_parity, 0);
        assert_eq!(
            vector_bytes,
            gpu.read_texture(
                canvas.vector_texture(),
                canvas_grid,
                GESTURE_CANVAS_VECTOR_TEXEL_BYTES
            )
        );
        assert_eq!(
            gate_bytes,
            gpu.read_texture(
                canvas.gate_texture(),
                canvas_grid,
                GESTURE_CANVAS_GATE_TEXEL_BYTES
            )
        );

        let mut decaying = GestureCanvasState::new(canvas_grid, authored_params(0.35, 0.5, 0.9))
            .expect("canvas state");
        decaying
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &events,
                evaluated_params: None,
            })
            .expect("staged");
        decaying.commit_staged();
        let mut decay_resources = GestureCanvasResources::prepare(
            &gpu.device,
            &gpu.queue,
            &[GestureCanvasRequest::new(canvas_grid)],
            limits(),
        )
        .expect("prepared canvas");
        let decay_canvas = decay_resources.canvas_mut(0).expect("canvas");
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture canvas seed"),
                });
            let plan = GestureCanvasFramePlan {
                decay_ticks: 0,
                decay_clamped: false,
                applied_samples: 2,
                inert_samples: 0,
                held: false,
            };
            let samples: Vec<GestureEtchSample> = events
                .iter()
                .map(|e| GestureEtchSample::from_event(*e))
                .collect();
            decay_canvas
                .encode(&gpu.queue, &mut encoder, decaying.params(), plan, &samples)
                .expect("seed encode");
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
        let report = gpu.advance(
            &mut decaying,
            decay_canvas,
            GestureCanvasFrameInput {
                reference_tick: 6,
                program_advances: true,
                events: &[],
                evaluated_params: None,
            },
        );
        assert_eq!(report.executed_passes, 1);
        assert_eq!(report.decay_ticks, 6);
        assert_eq!(report.applied_samples, 0);
        assert_matches_reference(
            &gpu.read_field(decay_canvas),
            decaying.field(),
            3,
            "decay only",
        );

        // A hard reset clears both parities and returns the committed parity to
        // zero, so the device memory and the reset CPU state agree.
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture canvas clear encoder"),
                });
            decay_canvas.encode_clear(&mut encoder);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
        decaying.reset_for(crate::gesture_canvas::GestureCanvasResetCause::PatchGeneration);
        assert_eq!(decay_canvas.committed_parity(), 0);
        assert_matches_reference(
            &gpu.read_field(decay_canvas),
            decaying.field(),
            0,
            "hard reset",
        );

        // A plan and a sample list that disagree are refused rather than
        // encoded with whichever count happened to be shorter.
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gesture canvas mismatch encoder"),
            });
        assert_eq!(
            decay_canvas.encode(
                &gpu.queue,
                &mut encoder,
                decaying.params(),
                GestureCanvasFramePlan {
                    decay_ticks: 0,
                    decay_clamped: false,
                    applied_samples: 2,
                    inert_samples: 0,
                    held: false,
                },
                &[],
            ),
            Err(GestureCanvasGpuError::SampleCountMismatch {
                planned: 2,
                supplied: 0,
            })
        );
    }

    /// The presented donor is the routable half of this whole subsystem: it is
    /// the only surface anything outside the canvas ever binds. Two laws are
    /// proven on the device here rather than argued in prose.
    ///
    /// * The device presentation reproduces `present_displace_donor` cell for
    ///   cell — the CPU reference frozen by the previous stage is the contract,
    ///   and `fs_present` is its implementation, not a second convention.
    /// * A clear reaches the presented image too. A reset that rewound only the
    ///   working parities would leave the previously etched donor bound and
    ///   readable, which is exactly the stale state a reset exists to remove.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_the_presented_donor_reproduces_the_cpu_reference_and_a_clear_reaches_it() {
        use crate::gesture_canvas::{decode_displace_donor, present_displace_donor};

        let gpu = GpuHarness::new();
        let canvas_grid = grid(8, 8);
        let params = authored_params(0.5, 0.8, 1.0);
        let mut resources = GestureCanvasResources::prepare(
            &gpu.device,
            &gpu.queue,
            &[GestureCanvasRequest::new(canvas_grid)],
            limits(),
        )
        .expect("prepared canvas");
        let mut state = GestureCanvasState::new(canvas_grid, params).expect("canvas state");

        // A fresh canvas is initialized, not merely allocated: every presented
        // texel decodes to exactly zero before anything etches.
        let blank = gpu.read_presented(resources.canvas(0).expect("canvas"));
        assert!(
            blank
                .iter()
                .all(|texel| decode_displace_donor(*texel) == [0.0, 0.0]),
            "a freshly prepared canvas presented a non-zero donor"
        );

        let events = [
            event(0, GestureMode::Push, [0.3, 0.35], 1.0, [1.0, 0.0]),
            event(1, GestureMode::Curl, [0.7, 0.6], 0.8, [0.0, 1.0]),
        ];
        gpu.advance(
            &mut state,
            resources.canvas_mut(0).expect("canvas"),
            GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &events,
                evaluated_params: None,
            },
        );

        let presented = gpu.read_presented(resources.canvas(0).expect("canvas"));
        let mut etched_cells = 0_usize;
        for y in 0..canvas_grid.height {
            for x in 0..canvas_grid.width {
                let index = canvas_grid.index(x, y).expect("cell index");
                let reference = state
                    .field()
                    .present_displace_donor(x, y)
                    .expect("reference donor");
                if reference[3] > 0.0 {
                    etched_cells += 1;
                }
                for channel in 0..4 {
                    assert!(
                        (presented[index][channel] - reference[channel]).abs() <= VECTOR_TOLERANCE,
                        "cell ({x},{y}) channel {channel}: device {} vs reference {}",
                        presented[index][channel],
                        reference[channel]
                    );
                }
                // And the decode the consumer actually performs agrees too, so
                // the claim is about displacement rather than about storage.
                let device_vector = decode_displace_donor(presented[index]);
                let reference_vector = decode_displace_donor(present_displace_donor(
                    state.field().cell(x, y).expect("cell"),
                ));
                assert!(
                    (device_vector[0] - reference_vector[0]).abs() <= VECTOR_TOLERANCE
                        && (device_vector[1] - reference_vector[1]).abs() <= VECTOR_TOLERANCE
                );
            }
        }
        assert!(etched_cells > 0, "the fixture etched nothing to present");
        assert!(
            presented
                .iter()
                .any(|texel| decode_displace_donor(*texel) != [0.0, 0.0]),
            "the presented donor decodes to zero everywhere after an etch"
        );

        // Blue is unused and is written as an explicit zero, never left to
        // whatever the attachment happened to hold.
        assert!(presented.iter().all(|texel| texel[2] == 0.0));

        // A reset clears the presented image with the working parities.
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture canvas reset encoder"),
                });
            resources.encode_clear(&mut encoder);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
        state.reset_for(crate::gesture_canvas::GestureCanvasResetCause::PatchGeneration);
        let cleared = gpu.read_presented(resources.canvas(0).expect("canvas"));
        assert_eq!(
            cleared, blank,
            "a reset left a stale etched donor image bound"
        );
        assert_eq!(resources.canvas(0).expect("canvas").committed_parity(), 0);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_gesture_canvas_overlapping_strokes_compose_in_recorded_order_on_the_device_too() {
        let gpu = GpuHarness::new();
        let canvas_grid = grid(8, 8);
        let params = authored_params(0.5, 0.6, 1.0);
        let first = event(0, GestureMode::Push, [0.45, 0.5], 1.0, [1.0, 0.0]);
        let second = event(1, GestureMode::Push, [0.55, 0.5], 1.0, [0.0, 1.0]);

        let render = |events: [GestureEvent; 2]| {
            let mut resources = GestureCanvasResources::prepare(
                &gpu.device,
                &gpu.queue,
                &[GestureCanvasRequest::new(canvas_grid)],
                limits(),
            )
            .expect("prepared canvas");
            let mut state = GestureCanvasState::new(canvas_grid, params).expect("canvas state");
            let canvas = resources.canvas_mut(0).expect("canvas");
            let report = gpu.advance(
                &mut state,
                canvas,
                GestureCanvasFrameInput {
                    reference_tick: 0,
                    program_advances: true,
                    events: &events,
                    evaluated_params: None,
                },
            );
            assert_eq!(report.executed_passes, 2);
            let readback = gpu.read_field(canvas);
            assert_matches_reference(&readback, state.field(), 2, "ordered overlap");
            readback
        };

        let forward = render([first, second]);
        let reversed = render([second, first]);
        let overlap = canvas_grid.index(4, 4).expect("overlap cell");
        let separation = (forward[overlap].0 - reversed[overlap].0).abs()
            + (forward[overlap].1 - reversed[overlap].1).abs();
        assert!(
            separation > 0.05,
            "reordering two overlapping strokes changed the device field by only {separation}"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_gesture_canvas_live_and_export_reference_ticks_read_back_the_same_field() {
        let gpu = GpuHarness::new();
        let canvas_grid = grid(8, 8);
        let params = authored_params(0.4, 0.5, 0.97);
        let fps = 30_u32;
        let frames = 12_u64;

        // One recorded track, addressed on the 30 Hz reference and replayed
        // twice: once against the live accepted-frame accumulator and once
        // against the offline rounded rational map. Neither derivation reads
        // wall time, and both must land on the same addresses.
        let mut track = GestureTrack::default();
        for (tick, stroke, phase, mode, x) in [
            (0_u64, 0_u8, GesturePhase::Begin, GestureMode::Push, 0.2_f32),
            (2, 0, GesturePhase::Move, GestureMode::Push, 0.35),
            (5, 1, GesturePhase::Begin, GestureMode::Curl, 0.5),
            (9, 1, GesturePhase::Move, GestureMode::Curl, 0.65),
        ] {
            let recorded = track
                .record_accepted(
                    tick,
                    GestureEvent::quantized(stroke, phase, mode, [x, 0.5], 1.0, [1.0, 0.0]),
                )
                .expect("well-formed recorded event");
            assert!(recorded, "the fixture track reached its cap");
        }

        let mut live_ticks = Vec::with_capacity(frames as usize);
        let mut recorder = GestureEventRecorder::default();
        for _ in 0..frames {
            live_ticks.push(recorder.reference_tick());
            recorder.record_accepted(1.0 / fps as f32, &[]);
        }
        let export_ticks: Vec<u64> = (0..frames)
            .map(|frame| export_temporal_reference_tick(frame, fps))
            .collect();
        assert_eq!(
            live_ticks, export_ticks,
            "the two reference-tick derivations disagreed before any rendering"
        );

        let render = |ticks: &[u64]| {
            let mut resources = GestureCanvasResources::prepare(
                &gpu.device,
                &gpu.queue,
                &[GestureCanvasRequest::new(canvas_grid)],
                limits(),
            )
            .expect("prepared canvas");
            let mut state = GestureCanvasState::new(canvas_grid, params).expect("canvas state");
            let canvas = resources.canvas_mut(0).expect("canvas");
            let mut replay = track.replay();
            for tick in ticks {
                let due = replay.events_due(u32::try_from(*tick).expect("relative tick"));
                gpu.advance(
                    &mut state,
                    canvas,
                    GestureCanvasFrameInput {
                        reference_tick: *tick,
                        program_advances: true,
                        events: due,
                        evaluated_params: None,
                    },
                );
            }
            let vectors = gpu.read_texture(
                canvas.vector_texture(),
                canvas_grid,
                GESTURE_CANVAS_VECTOR_TEXEL_BYTES,
            );
            let gates = gpu.read_texture(
                canvas.gate_texture(),
                canvas_grid,
                GESTURE_CANVAS_GATE_TEXEL_BYTES,
            );
            let reference = state.field().clone();
            (vectors, gates, reference)
        };

        let (live_vectors, live_gates, live_reference) = render(&live_ticks);
        let (export_vectors, export_gates, export_reference) = render(&export_ticks);
        assert_eq!(
            live_vectors, export_vectors,
            "live and export vector fields differ"
        );
        assert_eq!(
            live_gates, export_gates,
            "live and export gate fields differ"
        );
        assert_eq!(live_reference, export_reference);

        // And the device field still equals the independent CPU reference, so
        // the two paths agree with each other *and* with the model.
        let decoded: Vec<(f32, f32, f32, f32)> = live_vectors
            .chunks_exact(4)
            .zip(live_gates.chunks_exact(2))
            .map(|(vector, gate)| {
                (
                    f16_to_f32(u16::from_le_bytes([vector[0], vector[1]])),
                    f16_to_f32(u16::from_le_bytes([vector[2], vector[3]])),
                    f32::from(gate[0]) / 255.0,
                    f32::from(gate[1]) / 255.0,
                )
            })
            .collect();
        assert_matches_reference(&decoded, &live_reference, 16, "live and export parity");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_gesture_canvas_admission_rejects_every_over_budget_request_before_allocating() {
        let gpu = GpuHarness::new();
        let allocations = |resources: &GestureCanvasResources| resources.allocation_snapshot();

        // Three canvases, one over the active limit.
        let request = GestureCanvasRequest::new(grid(8, 8));
        assert_eq!(
            GestureCanvasResources::prepare(&gpu.device, &gpu.queue, &[request; 3], limits()).err(),
            Some(GestureCanvasGpuError::Plan(
                GestureCanvasError::TooManyCanvases { count: 3, limit: 2 }
            ))
        );

        // A narrowed aggregate ceiling refuses before any texture exists.
        let bytes = grid(64, 64).bytes().expect("bytes");
        let narrowed = limits()
            .bounded(GESTURE_CANVAS_MAX_BYTES, bytes * 2 - 1)
            .expect("narrowed");
        assert!(matches!(
            GestureCanvasResources::prepare(
                &gpu.device,
                &gpu.queue,
                &[GestureCanvasRequest::new(grid(64, 64)); 2],
                narrowed
            ),
            Err(GestureCanvasGpuError::Plan(
                GestureCanvasError::AggregateBytes { .. }
            ))
        ));

        // Two canvases at the ceiling are admitted, allocate exactly twice the
        // single-canvas resources, and reconcile.
        let resources = GestureCanvasResources::prepare(
            &gpu.device,
            &gpu.queue,
            &[request; 2],
            limits()
                .bounded(GESTURE_CANVAS_MAX_BYTES, GESTURE_CANVAS_AGGREGATE_MAX_BYTES)
                .expect("ceilings"),
        )
        .expect("prepared");
        assert_eq!(
            allocations(&resources),
            GestureCanvasAllocationSnapshot {
                textures: 10,
                buffers: 2,
                bind_groups: 4,
                pipelines: 4,
            }
        );
        let ledger = resources.ledger().expect("ledger");
        assert_eq!(ledger.bytes_per_cell, GESTURE_CANVAS_BYTES_PER_CELL);
        assert_eq!(ledger.uniform_stride, GESTURE_CANVAS_UNIFORM_STRIDE);
        assert_eq!(resources.plan().reconcile(ledger), Ok(()));
    }
}
