//! The B8 melting-edge stage executor: the master melt over the program's
//! own coverage, on the one slot-0 seam every audience path shares. Live
//! LegacyExact, live Advanced, the selective-VHS path, and export all
//! converge on composite slot 0 immediately before the B4 display stage and
//! `render_opaque_output`, so this single implementation and single shader
//! serve all four — the `encode_opaque_output` precedent, exactly the B4
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
//! CPU reference: `mixing_boundary.rs`, followed expression for expression
//! by `melting_edge.wgsl`.

use crate::effects::params::TEMPORAL_REFERENCE_FPS;
use crate::mixing_boundary::{MeltGpuUniforms, MeltParams};

const MELT_UNIFORM_BYTES: u64 = 48;

struct MeltSurfaces {
    history_texture: wgpu::Texture,
    /// slot 0 + history, the melt pass's one bind group.
    melt_group: wgpu::BindGroup,
}

pub struct MeltingEdgeGpu {
    melt_pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dimensions: [u32; 2],
    surfaces: Option<MeltSurfaces>,
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
        let melt_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Melting edge pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_melt"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_melt"),
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
            history_valid: false,
            last_store_ticks: 0,
            tick_accumulator: 0.0,
            total_ticks: 0,
        }
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

    fn read_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<[u8; 4]> {
        let padded_row = (SIZE * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Melt fixture staging"),
            size: u64::from(padded_row * SIZE),
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
        let mut pixels = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            let row = &mapped[(y * padded_row) as usize..];
            for x in 0..SIZE {
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
