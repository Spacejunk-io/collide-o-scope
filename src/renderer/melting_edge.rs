//! The B8 melting-edge stage executor: the master melt over the program's
//! own coverage, on the one slot-0 seam every audience path shares. Live
//! LegacyExact, live Advanced, and export all converge on composite slot 0
//! immediately before the B4 display stage, `render_opaque_output`, and the
//! final-program VHS finish, so this single implementation and single shader
//! serve all three — the `encode_opaque_output` precedent, exactly the B4
//! shape.
//!
//! The stage's matte is the composite's alpha channel, so every coverage
//! boundary in the program (static key alpha, cellular gap, group matte)
//! melts through this one mechanism. Its history — the stage's own previous
//! output — is one retained slot-format surface advanced by `copy_texture`
//! once per 30 Hz reference tick on the stage's own rational accumulator
//! (the B4 held-field law), so live at any fps and export at any fps creep
//! at the same rate and Pause holds the trail still. The surface is lazily
//! allocated on the first armed frame, retained after, and invalidated
//! (never freed) on disarm so a re-arm cannot resurrect a stale trail. The
//! trail is program memory on the temporal-feedback precedent — blackout
//! blacks the audience without erasing it, exactly as the temporal ring
//! survives blackout — while patch load and resize rebuild the stage.
//!
//! The monitor-only band source is a fixed 128×72 `Rgba8Unorm` texture
//! (36,864 nominal texel bytes); its shader module, pipeline layout,
//! pipeline, surface, view, and bind group are all created on the first
//! explicit one-shot request. It samples slot 0 before the melt pass, writes the same clamped
//! band/creep law as grayscale with opaque alpha, and exposes its view only
//! for that creative frame. A request while melt is inactive publishes
//! opaque zero. Disarm revokes pending and published validity; resize builds
//! a fresh stage with neither allocation nor validity. Unrequested frames
//! execute the historical creative path with no diagnostic pass.
//!
//! CPU reference: `mixing_boundary.rs`, followed expression for expression
//! by `melting_edge.wgsl`.

use crate::effects::params::TEMPORAL_REFERENCE_FPS;
use crate::mixing_boundary::{MeltGpuUniforms, MeltParams};

const MELT_UNIFORM_BYTES: u64 = 48;
pub const MELT_DIAGNOSTIC_WIDTH: u32 = 128;
pub const MELT_DIAGNOSTIC_HEIGHT: u32 = 72;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the exact diagnostic byte charge is exercised by allocation-ledger fixtures"
    )
)]
pub const MELT_DIAGNOSTIC_TEXTURE_BYTES: u64 =
    MELT_DIAGNOSTIC_WIDTH as u64 * MELT_DIAGNOSTIC_HEIGHT as u64 * 4;

/// Exact lazy-object snapshot and retained-texture charge for the stage. The
/// diagnostic texture is a fixed RGBA8 surface (36,864 bytes) and every
/// diagnostic-only object appears only after an explicit request. Backend-
/// private alignment and metadata are deliberately outside this nominal
/// texel-byte ledger, as they are for the existing history.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the exact diagnostic allocation ledger is exercised by focused fixtures"
    )
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MeltingEdgeAllocationSnapshot {
    pub history_textures: u64,
    pub history_texture_bytes: u64,
    pub diagnostic_textures: u64,
    pub diagnostic_texture_views: u64,
    pub diagnostic_bind_groups: u64,
    pub diagnostic_shader_modules: u64,
    pub diagnostic_pipeline_layouts: u64,
    pub diagnostic_pipelines: u64,
    pub diagnostic_texture_bytes: u64,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the exact diagnostic allocation ledger is exercised by focused fixtures"
    )
)]
impl MeltingEdgeAllocationSnapshot {
    pub const fn total_texture_bytes(self) -> u64 {
        self.history_texture_bytes + self.diagnostic_texture_bytes
    }
}

#[derive(Debug, Default)]
struct MeltDiagnosticState {
    requested: bool,
    valid: bool,
}

impl MeltDiagnosticState {
    fn request(&mut self) {
        self.requested = true;
    }

    /// Start one creative frame. A request is one-shot and the previous
    /// frame's validity is revoked before any new diagnostic work is encoded.
    fn begin_frame(&mut self) -> bool {
        self.valid = false;
        std::mem::take(&mut self.requested)
    }

    fn publish(&mut self) {
        self.valid = true;
    }

    fn disarm(&mut self) {
        self.requested = false;
        self.valid = false;
    }
}

struct MeltSurfaces {
    history_texture: wgpu::Texture,
    /// slot 0 + history, the melt pass's one bind group.
    melt_group: wgpu::BindGroup,
}

struct MeltDiagnosticSurface {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The exact pre-melt slot-0 view plus the shared uniform and sampler.
    group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the exact diagnostic allocation ledger is exercised by focused fixtures"
    )
)]
fn melting_edge_allocation_snapshot_for(
    dimensions: [u32; 2],
    history_allocated: bool,
    diagnostic_allocated: bool,
) -> MeltingEdgeAllocationSnapshot {
    let history_texture_bytes = u64::from(dimensions[0].max(1))
        .saturating_mul(u64::from(dimensions[1].max(1)))
        .saturating_mul(4)
        .saturating_mul(u64::from(history_allocated));
    MeltingEdgeAllocationSnapshot {
        history_textures: u64::from(history_allocated),
        history_texture_bytes,
        diagnostic_textures: u64::from(diagnostic_allocated),
        diagnostic_texture_views: u64::from(diagnostic_allocated),
        diagnostic_bind_groups: u64::from(diagnostic_allocated),
        diagnostic_shader_modules: u64::from(diagnostic_allocated),
        diagnostic_pipeline_layouts: u64::from(diagnostic_allocated),
        diagnostic_pipelines: u64::from(diagnostic_allocated),
        diagnostic_texture_bytes: MELT_DIAGNOSTIC_TEXTURE_BYTES
            .saturating_mul(u64::from(diagnostic_allocated)),
    }
}

pub struct MeltingEdgeGpu {
    melt_pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dimensions: [u32; 2],
    surfaces: Option<MeltSurfaces>,
    diagnostic_surface: Option<MeltDiagnosticSurface>,
    diagnostic_state: MeltDiagnosticState,
    history_valid: bool,
    last_store_ticks: u64,
    /// The stage's own 30 Hz reference clock — the
    /// `history_ticks_for_delta` law, owned here because the Exact temporal
    /// state does not advance on Advanced frames while this stage runs on
    /// every path.
    tick_accumulator: f64,
    total_ticks: u64,
}

impl MeltingEdgeGpu {
    pub fn new(
        device: &wgpu::Device,
        composite_format: wgpu::TextureFormat,
        dimensions: [u32; 2],
    ) -> Self {
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Melting edge bind layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(MELT_UNIFORM_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Melting edge shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/melting_edge.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Melting edge pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let make_pipeline = |label: &str, entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_melt"),
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
        let melt_pipeline = make_pipeline("Melting edge pipeline", "fs_melt", composite_format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Melting edge sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Melting edge uniforms"),
            size: MELT_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            melt_pipeline,
            bind_layout,
            sampler,
            uniform,
            dimensions,
            surfaces: None,
            diagnostic_surface: None,
            diagnostic_state: MeltDiagnosticState::default(),
            history_valid: false,
            last_store_ticks: 0,
            tick_accumulator: 0.0,
            total_ticks: 0,
        }
    }

    /// Invalidate retained audience memory after the wet/dry Temporal
    /// partition changes. The texture remains allocated, but the next armed
    /// frame cannot sample pixels authored by a layer that has just moved to
    /// the dry overlay.
    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.last_store_ticks = self.total_ticks;
    }

    /// Request one 128×72 observation of the exact alpha field that will
    /// enter this frame's melt. Requests coalesce and are consumed by the
    /// next [`Self::encode`] call; ordinary frames allocate and encode no
    /// diagnostic resource.
    pub fn request_diagnostic_mask(&mut self) {
        self.diagnostic_state.request();
    }

    /// Revoke both a pending request and the last published validity when
    /// the monitor producer is disarmed. The lazily allocated surface may be
    /// retained, but its old bytes are unreachable until a new request is
    /// materialized at the melt seat.
    pub fn disarm_diagnostic_mask(&mut self) {
        self.diagnostic_state.disarm();
    }

    /// Whether the view names a mask materialized for the current creative
    /// frame. `encode` revokes this before consuming its one-shot request, so
    /// a missed cadence tick can never expose a stale frame as current.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "focused GPU fixtures assert current-frame diagnostic validity explicitly"
        )
    )]
    pub fn diagnostic_mask_valid(&self) -> bool {
        self.diagnostic_state.valid
    }

    /// Read-only monitor source. The view is exposed only while its current-
    /// frame validity is true; callers cannot accidentally reduce retained
    /// bytes after disarm or on a later unrequested frame.
    pub fn diagnostic_mask_view(&self) -> Option<&wgpu::TextureView> {
        self.diagnostic_state
            .valid
            .then(|| &self.diagnostic_surface.as_ref().expect("valid mask").view)
    }

    /// Exact diagnostic object and nominal texel allocation after lazy
    /// materialization. Byte totals exclude backend-private alignment and
    /// driver metadata.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the exact diagnostic allocation ledger is exercised by focused fixtures"
        )
    )]
    pub(crate) fn allocation_snapshot(&self) -> MeltingEdgeAllocationSnapshot {
        melting_edge_allocation_snapshot_for(
            self.dimensions,
            self.surfaces.is_some(),
            self.diagnostic_surface.is_some(),
        )
    }

    fn ensure_diagnostic_surface(
        &mut self,
        device: &wgpu::Device,
        composite_views: &[wgpu::TextureView; 3],
    ) {
        if self.diagnostic_surface.is_some() {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Melting edge diagnostic mask"),
            size: wgpu::Extent3d {
                width: MELT_DIAGNOSTIC_WIDTH,
                height: MELT_DIAGNOSTIC_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Every diagnostic-only GPU object is born at this same explicit
        // request boundary. An unrequested stage therefore owns no extra
        // shader module, pipeline layout, pipeline, surface, view, or group.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Melting edge diagnostic shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/melting_edge.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Melting edge diagnostic pipeline layout"),
            bind_group_layouts: &[Some(&self.bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Melting edge diagnostic pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_melt"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_melt_mask"),
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
        });
        // `fs_melt_mask` does not read binding 1. Bind slot 0 there as well
        // so the diagnostic shares the creative layout without allocating a
        // dummy history or aliasing its own render attachment as an input.
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Melting edge diagnostic group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(MELT_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.diagnostic_surface = Some(MeltDiagnosticSurface {
            _texture: texture,
            view,
            group,
            pipeline,
        });
    }

    /// Encode the diagnostic before the creative pass has any chance to
    /// overwrite slot 0. Inactive melt is a deliberate all-zero mask with
    /// opaque alpha; active melt runs the shader's exact band/creep law.
    fn materialize_diagnostic_mask(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        composite_views: &[wgpu::TextureView; 3],
        active: bool,
    ) {
        self.ensure_diagnostic_surface(device, composite_views);
        let surface = self.diagnostic_surface.as_ref().expect("ensured above");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Melting edge diagnostic pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface.view,
                resolve_target: None,
                ops: wgpu::Operations {
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
        if active {
            pass.set_pipeline(&surface.pipeline);
            pass.set_bind_group(0, &surface.group, &[]);
            pass.draw(0..3, 0..1);
        }
        drop(pass);
        self.diagnostic_state.publish();
    }

    fn ensure_surfaces(
        &mut self,
        device: &wgpu::Device,
        composite_format: wgpu::TextureFormat,
        composite_views: &[wgpu::TextureView; 3],
    ) {
        if self.surfaces.is_some() {
            return;
        }
        let [width, height] = self.dimensions;
        // The history matches the slot format byte for byte so the tick
        // update is one texture copy — the B4 held-field charge: 4 bytes a
        // pixel, cheaper than a working-format pair.
        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Melting edge history"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: composite_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_view = history_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let melt_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Melting edge group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(MELT_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.surfaces = Some(MeltSurfaces {
            history_texture,
            melt_group,
        });
    }

    /// Encode the stage for one frame. A dormant stage (no authored melt)
    /// encodes nothing, touches no surface, and invalidates the retained
    /// history so a later re-arm starts clean. Returns whether anything was
    /// encoded.
    #[allow(
        clippy::too_many_arguments,
        reason = "the stage names every borrowed frame input explicitly"
    )]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        composite_textures: &[wgpu::Texture; 3],
        composite_views: &[wgpu::TextureView; 3],
        composite_format: wgpu::TextureFormat,
        params: &MeltParams,
        delta_seconds: f32,
    ) -> bool {
        let diagnostic_requested = self.diagnostic_state.begin_frame();
        // The reference clock advances whether or not the stage is armed, so
        // arming mid-session lands on the same tick address either way.
        let dt = if delta_seconds.is_finite() {
            delta_seconds.max(0.0)
        } else {
            0.0
        };
        let reference_delta = 1.0 / f64::from(TEMPORAL_REFERENCE_FPS);
        self.tick_accumulator += f64::from(dt);
        let elapsed = (self.tick_accumulator / reference_delta).floor() as u64;
        if elapsed > 0 {
            self.tick_accumulator -= elapsed as f64 * reference_delta;
            self.total_ticks = self.total_ticks.saturating_add(elapsed);
        }
        let clean = params.sanitized();
        if diagnostic_requested {
            if clean.is_active() {
                let uniforms =
                    MeltGpuUniforms::from_parts(clean, self.dimensions, self.history_valid);
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
            }
            self.materialize_diagnostic_mask(device, encoder, composite_views, clean.is_active());
        }
        if !clean.is_active() {
            self.history_valid = false;
            return false;
        }
        self.ensure_surfaces(device, composite_format, composite_views);
        let uniforms = MeltGpuUniforms::from_parts(clean, self.dimensions, self.history_valid);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
        let surfaces = self.surfaces.as_ref().expect("ensured above");
        // 1. The melt pass reads slot 0 (and the history when armed and
        //    valid), ending in slot 2, then the result returns to slot 0 —
        //    the display-stage slot dance.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Melting edge pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[2],
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
        pass.set_pipeline(&self.melt_pipeline);
        pass.set_bind_group(0, &surfaces.melt_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        copy_full_frame(
            encoder,
            &composite_textures[2],
            &composite_textures[0],
            self.dimensions,
        );
        // 2. The history advances once per reference tick, from the melted
        //    output itself — the self-feed that makes the smear creep. An
        //    unarmed hold keeps no memory at all.
        if clean.is_armed() {
            if self.total_ticks != self.last_store_ticks || !self.history_valid {
                copy_full_frame(
                    encoder,
                    &composite_textures[0],
                    &surfaces.history_texture,
                    self.dimensions,
                );
                self.last_store_ticks = self.total_ticks;
                self.history_valid = true;
            }
        } else {
            self.history_valid = false;
        }
        true
    }
}

fn copy_full_frame(
    encoder: &mut wgpu::CommandEncoder,
    from: &wgpu::Texture,
    to: &wgpu::Texture,
    dimensions: [u32; 2],
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: from,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: to,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: dimensions[0].max(1),
            height: dimensions[1].max(1),
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: u32 = 32;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Melting edge test"),
            ..Default::default()
        }))
        .ok()
    }

    fn composites(device: &wgpu::Device) -> ([wgpu::Texture; 3], [wgpu::TextureView; 3]) {
        let textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Melt fixture composite {i}")),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let views = std::array::from_fn(|i| {
            textures[i].create_view(&wgpu::TextureViewDescriptor::default())
        });
        (textures, views)
    }

    /// A vertical coverage edge: columns left of `edge` carry the covered
    /// colour, the rest is transparent black.
    fn write_edge(queue: &wgpu::Queue, texture: &wgpu::Texture, edge: u32, rgba: [u8; 4]) {
        let mut bytes = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for _y in 0..SIZE {
            for x in 0..SIZE {
                if x < edge {
                    bytes.extend_from_slice(&rgba);
                } else {
                    bytes.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
    }

    fn read_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let padded_row = (width * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Melt fixture staging"),
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
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let row = &mapped[(y * padded_row) as usize..];
            pixels.extend_from_slice(&row[..(width * 4) as usize]);
        }
        drop(mapped);
        staging.unmap();
        pixels
    }

    fn read_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<[u8; 4]> {
        read_rgba8(device, queue, texture, SIZE, SIZE)
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect()
    }

    fn run_frame(
        stage: &mut MeltingEdgeGpu,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        composites: &([wgpu::Texture; 3], [wgpu::TextureView; 3]),
        params: &MeltParams,
        delta: f32,
    ) -> bool {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let encoded = stage.encode(
            device,
            queue,
            &mut encoder,
            &composites.0,
            &composites.1,
            FORMAT,
            params,
            delta,
        );
        queue.submit(std::iter::once(encoder.finish()));
        encoded
    }

    #[test]
    fn diagnostic_request_is_one_shot_and_disarm_revokes_every_validity_path() {
        let mut state = MeltDiagnosticState::default();
        assert!(!state.begin_frame());
        assert!(!state.valid);

        state.request();
        state.request();
        assert!(state.begin_frame(), "coalesced request was not consumed");
        assert!(!state.begin_frame(), "a request leaked into a second frame");
        state.publish();
        assert!(state.valid);

        state.request();
        state.disarm();
        assert!(!state.valid, "disarm retained a published mask");
        assert!(!state.begin_frame(), "disarm retained a pending request");
    }

    #[test]
    fn diagnostic_surface_has_one_exact_lazy_rgba8_charge() {
        let empty = melting_edge_allocation_snapshot_for([1_920, 1_080], false, false);
        assert_eq!(empty, MeltingEdgeAllocationSnapshot::default());

        let diagnostic = melting_edge_allocation_snapshot_for([1_920, 1_080], false, true);
        assert_eq!(
            diagnostic,
            MeltingEdgeAllocationSnapshot {
                history_textures: 0,
                history_texture_bytes: 0,
                diagnostic_textures: 1,
                diagnostic_texture_views: 1,
                diagnostic_bind_groups: 1,
                diagnostic_shader_modules: 1,
                diagnostic_pipeline_layouts: 1,
                diagnostic_pipelines: 1,
                diagnostic_texture_bytes: 128 * 72 * 4,
            }
        );
        assert_eq!(diagnostic.total_texture_bytes(), 36_864);

        let warmed = melting_edge_allocation_snapshot_for([1_920, 1_080], true, true);
        assert_eq!(warmed.history_textures, 1);
        assert_eq!(warmed.history_texture_bytes, 1_920 * 1_080 * 4);
        assert_eq!(warmed.diagnostic_texture_bytes, 36_864);
        assert_eq!(warmed.total_texture_bytes(), 8_294_400 + 36_864);
    }

    #[test]
    fn creative_melt_and_diagnostic_mask_share_one_shader_band_oracle() {
        let shader = include_str!("../shaders/melting_edge.wgsl").replace("\r\n", "\n");
        assert_eq!(shader.matches("fn melt_band_sample(").count(), 1);
        assert!(shader.contains(
            "fn fs_melt(@location(0) uv: vec2f) -> @location(0) vec4f {\n    let sample = melt_band_sample(uv);"
        ));
        assert!(shader.contains(
            "fn fs_melt_mask(@location(0) uv: vec2f) -> @location(0) vec4f {\n    let band = melt_band_sample(uv).band;\n    return vec4f(band, band, band, 1.0);"
        ));

        let source = include_str!("melting_edge.rs").replace("\r\n", "\n");
        assert!(source.contains("let diagnostic_requested = self.diagnostic_state.begin_frame();"));
        assert!(source.contains(
            "self.materialize_diagnostic_mask(device, encoder, composite_views, clean.is_active());"
        ));
    }

    /// The diagnostic is sampled from slot 0 before the creative pass. Its
    /// active bytes agree with the two CPU owner functions to one RGBA8 LSB;
    /// inactive melt is opaque zero; and an unrequested later frame revokes
    /// validity without allocating another surface.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_melt_diagnostic_mask_matches_the_cpu_band_and_creep_oracles() {
        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let composites = composites(&device);
        let mut stage = MeltingEdgeGpu::new(&device, FORMAT, [SIZE, SIZE]);
        assert_eq!(
            stage.allocation_snapshot(),
            MeltingEdgeAllocationSnapshot::default()
        );

        // An explicit observation of an inactive stage is truthful opaque
        // zero and does not wake the full-frame history allocation.
        write_edge(&queue, &composites.0[0], SIZE / 2, [255, 255, 255, 255]);
        stage.request_diagnostic_mask();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(!stage.encode(
            &device,
            &queue,
            &mut encoder,
            &composites.0,
            &composites.1,
            FORMAT,
            &MeltParams::default(),
            1.0 / 30.0,
        ));
        queue.submit(std::iter::once(encoder.finish()));
        assert!(stage.diagnostic_mask_valid());
        assert!(stage.diagnostic_mask_view().is_some());
        let inactive = read_rgba8(
            &device,
            &queue,
            &stage
                .diagnostic_surface
                .as_ref()
                .expect("allocated diagnostic")
                ._texture,
            MELT_DIAGNOSTIC_WIDTH,
            MELT_DIAGNOSTIC_HEIGHT,
        );
        assert!(
            inactive
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 0, 0, 255]),
            "inactive melt exposed a latent coverage edge"
        );
        assert_eq!(
            stage.allocation_snapshot(),
            MeltingEdgeAllocationSnapshot {
                history_textures: 0,
                history_texture_bytes: 0,
                diagnostic_textures: 1,
                diagnostic_texture_views: 1,
                diagnostic_bind_groups: 1,
                diagnostic_shader_modules: 1,
                diagnostic_pipeline_layouts: 1,
                diagnostic_pipelines: 1,
                diagnostic_texture_bytes: MELT_DIAGNOSTIC_TEXTURE_BYTES,
            }
        );

        // A later frame without a request cannot advertise the retained
        // bytes as current.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(!stage.encode(
            &device,
            &queue,
            &mut encoder,
            &composites.0,
            &composites.1,
            FORMAT,
            &MeltParams::default(),
            1.0 / 30.0,
        ));
        queue.submit(std::iter::once(encoder.finish()));
        assert!(!stage.diagnostic_mask_valid());
        assert!(stage.diagnostic_mask_view().is_none());

        let params = MeltParams {
            melt: 1.0,
            width: 0.2,
            hold: 0.0,
            swirl: 0.35,
            chroma: 0.0,
            creep: 0.65,
        };
        write_edge(&queue, &composites.0[0], SIZE / 2, [255, 255, 255, 255]);
        stage.request_diagnostic_mask();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(stage.encode(
            &device,
            &queue,
            &mut encoder,
            &composites.0,
            &composites.1,
            FORMAT,
            &params,
            1.0 / 30.0,
        ));
        queue.submit(std::iter::once(encoder.finish()));
        let actual = read_rgba8(
            &device,
            &queue,
            &stage
                .diagnostic_surface
                .as_ref()
                .expect("allocated diagnostic")
                ._texture,
            MELT_DIAGNOSTIC_WIDTH,
            MELT_DIAGNOSTIC_HEIGHT,
        );

        let edge_alpha = |uv_x: f32| {
            let position = uv_x.clamp(0.0, 1.0) * SIZE as f32 - 0.5;
            let base = position.floor();
            let fraction = position - base;
            let at = |index: i32| {
                let clamped = index.clamp(0, SIZE as i32 - 1) as u32;
                if clamped < SIZE / 2 {
                    1.0
                } else {
                    0.0
                }
            };
            at(base as i32) * (1.0 - fraction) + at(base as i32 + 1) * fraction
        };
        let radius = crate::mixing_boundary::melt_probe_radius(params.width);
        let aspect = SIZE as f32 / SIZE as f32;
        for y in 0..MELT_DIAGNOSTIC_HEIGHT {
            for x in 0..MELT_DIAGNOSTIC_WIDTH {
                let uv_x = (x as f32 + 0.5) / MELT_DIAGNOSTIC_WIDTH as f32;
                let probes = [
                    edge_alpha(uv_x - radius / aspect),
                    edge_alpha(uv_x + radius / aspect),
                    edge_alpha(uv_x),
                    edge_alpha(uv_x),
                ];
                let (band, _) = crate::mixing_boundary::melt_band_and_normal(probes, params.swirl);
                let expected =
                    crate::mixing_boundary::melt_creep_band(band, edge_alpha(uv_x), params.creep);
                let expected = (expected * 255.0).round() as u8;
                let offset = ((y * MELT_DIAGNOSTIC_WIDTH + x) * 4) as usize;
                let pixel = &actual[offset..offset + 4];
                for channel in &pixel[..3] {
                    assert!(
                        channel.abs_diff(expected) <= 1,
                        "mask ({x}, {y}) expected {expected}, got {pixel:?}"
                    );
                }
                assert_eq!(pixel[3], 255, "mask alpha at ({x}, {y})");
            }
        }

        let warmed = stage.allocation_snapshot();
        assert_eq!(warmed.diagnostic_textures, 1);
        assert_eq!(warmed.diagnostic_texture_bytes, 36_864);
        stage.request_diagnostic_mask();
        stage.disarm_diagnostic_mask();
        assert!(!stage.diagnostic_mask_valid());
        assert!(stage.diagnostic_mask_view().is_none());
        assert_eq!(stage.allocation_snapshot(), warmed);
    }

    /// The stage's pixel claims on a real adapter: a dormant stage encodes
    /// nothing and leaves slot 0 untouched; an active melt moves pixels only
    /// inside the band while the far field round-trips byte-exactly; the
    /// armed hold dissolves the stage's own history back into the band; and
    /// a coverage field with no boundary melts nothing — BENDR's own
    /// correctness case.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_melting_edge_drags_the_band_holds_history_and_needs_a_boundary() {
        let Some((device, queue)) = acquire_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let composites = composites(&device);
        let white = [255, 255, 255, 255];

        // 1. Dormant: no encode, slot 0 byte-identical.
        let mut stage = MeltingEdgeGpu::new(&device, FORMAT, [SIZE, SIZE]);
        write_edge(&queue, &composites.0[0], 16, white);
        let before = read_pixels(&device, &queue, &composites.0[0]);
        let encoded = run_frame(
            &mut stage,
            &device,
            &queue,
            &composites,
            &MeltParams::default(),
            1.0 / 30.0,
        );
        assert!(!encoded, "a dormant melt must not encode");
        assert_eq!(
            before,
            read_pixels(&device, &queue, &composites.0[0]),
            "a dormant melt must leave slot 0 untouched"
        );

        // 2. Active drag: the band moves pixels near the edge; the far field
        // survives the sample/render/copy round trip byte for byte.
        let active = MeltParams {
            melt: 2.0,
            width: 1.0,
            hold: 0.0,
            ..MeltParams::default()
        };
        assert!(run_frame(
            &mut stage,
            &device,
            &queue,
            &composites,
            &active,
            1.0 / 30.0,
        ));
        let melted = read_pixels(&device, &queue, &composites.0[0]);
        let mid_row = (SIZE / 2 * SIZE) as usize;
        assert_eq!(
            melted[mid_row + 2],
            white,
            "far field inside coverage is untouched"
        );
        assert_eq!(
            melted[mid_row + 29],
            [0, 0, 0, 0],
            "far field outside coverage is untouched"
        );
        let band_columns = 12..20;
        assert!(
            band_columns
                .clone()
                .any(|x| melted[mid_row + x] != before[mid_row + x]),
            "the band around the edge must move"
        );

        // 3. The armed hold keeps the stage's own history: a red frame is
        // stored, then a green frame melts — with hold the band still
        // carries red, without hold it cannot.
        let observed_red =
            |device: &wgpu::Device, queue: &wgpu::Queue, stage: &mut MeltingEdgeGpu, hold: f32| {
                let params = MeltParams {
                    melt: 2.0,
                    width: 1.0,
                    hold,
                    chroma: 0.0,
                    ..MeltParams::default()
                };
                write_edge(queue, &composites.0[0], 16, [255, 0, 0, 255]);
                // Two frames so the armed store publishes the red history.
                run_frame(stage, device, queue, &composites, &params, 1.0 / 30.0);
                write_edge(queue, &composites.0[0], 16, [0, 255, 0, 255]);
                run_frame(stage, device, queue, &composites, &params, 1.0 / 30.0);
                let pixels = read_pixels(device, queue, &composites.0[0]);
                (12..20)
                    .map(|x| u32::from(pixels[mid_row + x][0]))
                    .max()
                    .unwrap()
            };
        let mut held_stage = MeltingEdgeGpu::new(&device, FORMAT, [SIZE, SIZE]);
        let red_with_hold = observed_red(&device, &queue, &mut held_stage, 1.5);
        let mut dry_stage = MeltingEdgeGpu::new(&device, FORMAT, [SIZE, SIZE]);
        let red_without_hold = observed_red(&device, &queue, &mut dry_stage, 0.0);
        assert!(
            red_with_hold > red_without_hold + 32,
            "the hold must dissolve the red history into the band \
             (with: {red_with_hold}, without: {red_without_hold})"
        );

        // 4. No boundary, nothing happens: a fully covered flat frame has no
        // coverage disagreement, so the melt is the identity on it.
        let mut flat_stage = MeltingEdgeGpu::new(&device, FORMAT, [SIZE, SIZE]);
        write_edge(&queue, &composites.0[0], SIZE, white);
        let flat_before = read_pixels(&device, &queue, &composites.0[0]);
        assert!(run_frame(
            &mut flat_stage,
            &device,
            &queue,
            &composites,
            &active,
            1.0 / 30.0,
        ));
        assert_eq!(
            flat_before,
            read_pixels(&device, &queue, &composites.0[0]),
            "a boundary-free field must melt nothing"
        );
    }
}
