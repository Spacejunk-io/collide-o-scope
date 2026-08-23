//! The display-physics stage executor: the one seam every audience path
//! shares.
//!
//! Live LegacyExact, live Advanced, and export all converge on composite slot
//! 0 immediately before `render_opaque_output` and the final-program VHS
//! finish, so a single slot-0 stage here serves all three with one implementation
//! and one shader — export-identical by construction, the
//! `encode_opaque_output` precedent rather than the dual-implementation
//! temporal one.
//!
//! Surfaces are allocated lazily on the first armed frame and retained
//! after (BENDR's own rule: the persistence pair only exists once phosphor
//! is actually turned up), so a default session charges nothing; a warmed
//! armed frame allocates nothing. Blackout clears the phosphor accumulator
//! and the held field — a blacked-out audience must not retain a glowing
//! wake — and disarming the whole stage invalidates both, so a re-arm never
//! resurrects a stale trail.
//!
//! Pass order (the BENDR order, with its N-1 law kept):
//!   1. field pass (armed): slot 0 + held field → slot 2
//!   2. phosphor store (armed): slot 0 (pre-field signal) + ping → pong
//!   3. held-field update: copy slot 0 → field surface on a new reference
//!      tick
//!   4. display pass: (slot 2 | slot 0) + ping (the N-1 trail) → the other
//!      slot, ending in slot 0
//!   5. parities swap; the resolve consumes slot 0.

use crate::display_physics::{
    field_parity, judder_held, phosphor_decay_over_ticks, DisplayPhysicsGpuUniforms,
    DisplayPhysicsParams,
};
use crate::effects::params::TEMPORAL_REFERENCE_FPS;

const DISPLAY_UNIFORM_BYTES: u64 = 128;
/// The accumulator format: the trail outlives the 8-bit slot chain, so it
/// keeps linear headroom — the established feedback shape at `Rgba16Float`,
/// explicitly not a second history ring.
const PHOSPHOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct DisplaySurfaces {
    field_texture: wgpu::Texture,
    /// Views keep their textures alive; only the views are ever read.
    phosphor_views: [wgpu::TextureView; 2],
    /// slot 0 + held field.
    field_group: wgpu::BindGroup,
    /// slot 0 + phosphor ping, per parity.
    store_groups: [wgpu::BindGroup; 2],
    /// slot 0 + phosphor ping, per parity (the fields-off display input).
    model_from_slot0: [wgpu::BindGroup; 2],
    /// slot 2 + phosphor ping, per parity (the fields-on display input).
    model_from_slot2: [wgpu::BindGroup; 2],
}

pub struct DisplayPhysicsGpu {
    field_pipeline: wgpu::RenderPipeline,
    model_pipeline: wgpu::RenderPipeline,
    store_pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    dimensions: [u32; 2],
    composite_format: wgpu::TextureFormat,
    surfaces: Option<DisplaySurfaces>,
    field_valid: bool,
    phosphor_valid: bool,
    phosphor_parity: usize,
    last_field_ticks: u64,
    /// The stage's own 30 Hz reference clock — the same rational-accumulator
    /// law `TemporalState::history_ticks_for_delta` uses, owned here because
    /// the Exact temporal state does not advance on Advanced frames while
    /// this stage runs on every path. Fed by the program-advancing delta, so
    /// live at any fps and export at any fps address the same ticks.
    tick_accumulator: f64,
    total_ticks: u64,
}

impl DisplayPhysicsGpu {
    pub fn new(
        device: &wgpu::Device,
        composite_format: wgpu::TextureFormat,
        dimensions: [u32; 2],
    ) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display physics bind layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(DISPLAY_UNIFORM_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display physics shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/display_physics.wgsl").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Display physics pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let make_pipeline = |label: &str, entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_display"),
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
        let field_pipeline = make_pipeline(
            "Display physics field pipeline",
            "fs_display_field",
            composite_format,
        );
        let model_pipeline = make_pipeline(
            "Display physics model pipeline",
            "fs_display_model",
            composite_format,
        );
        let store_pipeline = make_pipeline(
            "Display physics store pipeline",
            "fs_display_store",
            PHOSPHOR_FORMAT,
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Display physics uniforms"),
            size: DISPLAY_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            field_pipeline,
            model_pipeline,
            store_pipeline,
            bind_layout,
            uniform,
            dimensions,
            composite_format,
            surfaces: None,
            field_valid: false,
            phosphor_valid: false,
            phosphor_parity: 0,
            last_field_ticks: 0,
            tick_accumulator: 0.0,
            total_ticks: 0,
        }
    }

    /// Revoke the held-field and phosphor memories when the Temporal wet/dry
    /// membership changes. Validity gates make the retained texture bytes
    /// unreachable, so this needs no full-frame clear or new allocation.
    pub fn invalidate_memory(&mut self) {
        self.field_valid = false;
        self.phosphor_valid = false;
    }

    fn ensure_surfaces(&mut self, device: &wgpu::Device, composite_views: &[wgpu::TextureView; 3]) {
        if self.surfaces.is_some() {
            return;
        }
        let [width, height] = self.dimensions;
        let make_texture = |label: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        // The held field matches the slot format byte for byte so the tick
        // update is one texture copy — the spec's RGBA8 charge, and the
        // signal domain interlace actually lives in.
        let field_texture = make_texture("Display physics held field", self.composite_format);
        let field_view = field_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let phosphor_textures = [
            make_texture("Display physics phosphor ping", PHOSPHOR_FORMAT),
            make_texture("Display physics phosphor pong", PHOSPHOR_FORMAT),
        ];
        let phosphor_views = [
            phosphor_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            phosphor_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let group = |label: &str, input: &wgpu::TextureView, aux: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(aux),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniform,
                            offset: 0,
                            size: std::num::NonZeroU64::new(DISPLAY_UNIFORM_BYTES),
                        }),
                    },
                ],
            })
        };
        self.surfaces = Some(DisplaySurfaces {
            field_group: group("Display field group", &composite_views[0], &field_view),
            store_groups: [
                group(
                    "Display store group A",
                    &composite_views[0],
                    &phosphor_views[0],
                ),
                group(
                    "Display store group B",
                    &composite_views[0],
                    &phosphor_views[1],
                ),
            ],
            model_from_slot0: [
                group(
                    "Display model group 0A",
                    &composite_views[0],
                    &phosphor_views[0],
                ),
                group(
                    "Display model group 0B",
                    &composite_views[0],
                    &phosphor_views[1],
                ),
            ],
            model_from_slot2: [
                group(
                    "Display model group 2A",
                    &composite_views[2],
                    &phosphor_views[0],
                ),
                group(
                    "Display model group 2B",
                    &composite_views[2],
                    &phosphor_views[1],
                ),
            ],
            field_texture,
            phosphor_views,
        });
    }

    /// Encode the whole stage for one frame. A dormant stage (all three
    /// sub-blocks off) encodes nothing, touches no surface, and invalidates
    /// the retained memories so a later re-arm starts clean rather than
    /// resurrecting a stale trail. Returns whether anything was encoded.
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
        params: &DisplayPhysicsParams,
        delta_seconds: f32,
    ) -> bool {
        // The reference clock advances whether or not the stage is armed, so
        // arming mid-session lands on the same field address either way.
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
        if !clean.stage_active() {
            self.field_valid = false;
            self.phosphor_valid = false;
            return false;
        }
        self.ensure_surfaces(device, composite_views);
        let fields_on = clean.fields_active();
        let phosphor_on = clean.phosphor_active();
        let total_reference_ticks = self.total_ticks;
        let parity = field_parity(total_reference_ticks, clean.il_order);
        // The multiplicative rate law: per-tick decay exponentiated by the
        // fractional reference ticks this frame spans. Live passes elapsed
        // dt, export passes frame-index-derived dt — the temporal law.
        let decay = phosphor_decay_over_ticks(clean.phosphor_decay(), dt * TEMPORAL_REFERENCE_FPS);
        let uniforms = DisplayPhysicsGpuUniforms::from_parts(
            &clean,
            self.dimensions,
            parity,
            judder_held(total_reference_ticks),
            decay,
            self.field_valid,
            self.phosphor_valid,
        );
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
        let surfaces = self.surfaces.as_ref().expect("ensured above");
        let ping = self.phosphor_parity & 1;
        // 1. The field domain recombines against the held field, into slot 2.
        if fields_on {
            encode_fullscreen(
                encoder,
                "Display physics field pass",
                &self.field_pipeline,
                &surfaces.field_group,
                &composite_views[2],
            );
        }
        // 2. The phosphor store reads the pre-field signal (BENDR's law) and
        //    the ping trail, writing the pong trail.
        if phosphor_on {
            encode_fullscreen(
                encoder,
                "Display physics store pass",
                &self.store_pipeline,
                &surfaces.store_groups[ping],
                &surfaces.phosphor_views[1 - ping],
            );
        }
        // 3. The held field advances once per reference tick, from the
        //    pre-field signal, after the field pass has read the old one.
        if fields_on && (total_reference_ticks != self.last_field_ticks || !self.field_valid) {
            copy_full_frame(
                encoder,
                &composite_textures[0],
                &surfaces.field_texture,
                self.dimensions,
            );
            self.last_field_ticks = total_reference_ticks;
            self.field_valid = true;
        }
        // 4. The display pass reads the post-field image and the N-1 trail
        //    (the ping the store did NOT write), ending in slot 0.
        if fields_on {
            encode_fullscreen(
                encoder,
                "Display physics model pass",
                &self.model_pipeline,
                &surfaces.model_from_slot2[ping],
                &composite_views[0],
            );
        } else {
            encode_fullscreen(
                encoder,
                "Display physics model pass",
                &self.model_pipeline,
                &surfaces.model_from_slot0[ping],
                &composite_views[2],
            );
            copy_full_frame(
                encoder,
                &composite_textures[2],
                &composite_textures[0],
                self.dimensions,
            );
        }
        if phosphor_on {
            self.phosphor_parity ^= 1;
            self.phosphor_valid = true;
        }
        true
    }

    /// Blackout clears the glowing wake and the held field. Cheap when the
    /// stage never armed: no surfaces, nothing to clear.
    pub fn clear_for_blackout(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.field_valid = false;
        self.phosphor_valid = false;
        let Some(surfaces) = &self.surfaces else {
            return;
        };
        for view in &surfaces.phosphor_views {
            clear_view(encoder, "Display physics blackout clear", view);
        }
    }
}

fn encode_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn clear_view(encoder: &mut wgpu::CommandEncoder, label: &str, view: &wgpu::TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: u32 = 8;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Display physics test"),
            ..Default::default()
        }))
        .ok()
    }

    fn composites(device: &wgpu::Device) -> ([wgpu::Texture; 3], [wgpu::TextureView; 3]) {
        let textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Display fixture composite {i}")),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

    fn write_flat(queue: &wgpu::Queue, texture: &wgpu::Texture, rgba: [u8; 4]) {
        let bytes: Vec<u8> = rgba
            .into_iter()
            .cycle()
            .take((SIZE * SIZE * 4) as usize)
            .collect();
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

    fn read_slot0(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<[u8; 4]> {
        let padded_row = (SIZE * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Display fixture staging"),
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

    fn encode_frame(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stage: &mut DisplayPhysicsGpu,
        textures: &[wgpu::Texture; 3],
        views: &[wgpu::TextureView; 3],
        params: &DisplayPhysicsParams,
        dt: f32,
    ) -> bool {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let encoded = stage.encode(device, queue, &mut encoder, textures, views, params, dt);
        queue.submit(std::iter::once(encoder.finish()));
        encoded
    }

    fn srgb_to_linear(byte: u8) -> f32 {
        let value = byte as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    /// The B4 physical-GPU claims in one fixture: a dormant stage is a real
    /// delegation (slot 0 byte-identical, no surfaces), the weave comb is
    /// two genuine moments in one frame, the phosphor trail follows the
    /// closed-form per-primary decay with its one-frame store lag and the
    /// P22 ordering, and blackout clears the glowing wake.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_display_physics_follows_the_cpu_laws_and_blackout_clears_the_wake() {
        let Some((device, queue)) = acquire_device() else {
            panic!("GPU adapter required");
        };
        let (textures, views) = composites(&device);
        let mut stage =
            DisplayPhysicsGpu::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, [SIZE, SIZE]);

        // A dormant stage encodes nothing and touches nothing.
        write_flat(&queue, &textures[0], [120, 60, 200, 255]);
        let before = read_slot0(&device, &queue, &textures[0]);
        let encoded = encode_frame(
            &device,
            &queue,
            &mut stage,
            &textures,
            &views,
            &DisplayPhysicsParams::default(),
            1.0 / 30.0,
        );
        assert!(!encoded, "the exact-off default must not encode");
        assert!(
            stage.surfaces.is_none(),
            "a dormant stage allocates nothing"
        );
        assert_eq!(before, read_slot0(&device, &queue, &textures[0]));

        // Weave: two moments in one frame. Tick one draws red (held as the
        // previous field), tick two draws green; the output rows alternate.
        let weave = DisplayPhysicsParams {
            il_amount: 1.0,
            il_twitter: 0.0,
            ..DisplayPhysicsParams::default()
        };
        write_flat(&queue, &textures[0], [255, 0, 0, 255]);
        assert!(encode_frame(
            &device,
            &queue,
            &mut stage,
            &textures,
            &views,
            &weave,
            1.0 / 30.0
        ));
        write_flat(&queue, &textures[0], [0, 255, 0, 255]);
        assert!(encode_frame(
            &device,
            &queue,
            &mut stage,
            &textures,
            &views,
            &weave,
            1.0 / 30.0
        ));
        let combed = read_slot0(&device, &queue, &textures[0]);
        let row = |y: u32| combed[(y * SIZE) as usize];
        let red_rows = (0..SIZE)
            .filter(|y| row(*y)[0] > 200 && row(*y)[1] < 60)
            .count();
        let green_rows = (0..SIZE)
            .filter(|y| row(*y)[1] > 200 && row(*y)[0] < 60)
            .count();
        assert_eq!(
            (red_rows, green_rows),
            (SIZE as usize / 2, SIZE as usize / 2),
            "the comb interleaves two genuine moments"
        );

        // Phosphor: a white impulse decays per the closed form, one frame
        // behind the store, green outlasting red outlasting blue.
        let mut stage =
            DisplayPhysicsGpu::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, [SIZE, SIZE]);
        let phosphor = DisplayPhysicsParams {
            phosphor: 0.9,
            phos_r: 0.86,
            phos_g: 1.0,
            phos_b: 0.66,
            ..DisplayPhysicsParams::default()
        };
        let decay = phosphor.phosphor_decay();
        write_flat(&queue, &textures[0], [255, 255, 255, 255]);
        assert!(encode_frame(
            &device,
            &queue,
            &mut stage,
            &textures,
            &views,
            &phosphor,
            1.0 / 30.0
        ));
        // Three black frames at exactly one reference tick each. The
        // displayed trail at black frame m is decay^(m-1): the store decayed
        // it m-1 times by the time the display reads it.
        for frame in 1..=3_u32 {
            write_flat(&queue, &textures[0], [0, 0, 0, 255]);
            assert!(encode_frame(
                &device,
                &queue,
                &mut stage,
                &textures,
                &views,
                &phosphor,
                1.0 / 30.0
            ));
            let trail = read_slot0(&device, &queue, &textures[0])[0];
            let expected: Vec<f32> = decay.iter().map(|k| k.powi(frame as i32 - 1)).collect();
            for channel in 0..3 {
                let observed = srgb_to_linear(trail[channel]);
                assert!(
                    (observed - expected[channel]).abs() < 0.02,
                    "frame {frame} channel {channel}: observed {observed}, expected {}",
                    expected[channel]
                );
            }
        }
        // P22 ordering after the trail has decayed: green > red > blue.
        let trail = read_slot0(&device, &queue, &textures[0])[0];
        assert!(trail[1] > trail[0] && trail[0] > trail[2]);

        // Blackout clears the wake: the next black frame shows no trail.
        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            stage.clear_for_blackout(&mut encoder);
            queue.submit(std::iter::once(encoder.finish()));
        }
        write_flat(&queue, &textures[0], [0, 0, 0, 255]);
        assert!(encode_frame(
            &device,
            &queue,
            &mut stage,
            &textures,
            &views,
            &phosphor,
            1.0 / 30.0
        ));
        let dark = read_slot0(&device, &queue, &textures[0])[0];
        assert!(
            dark[0] < 8 && dark[1] < 8 && dark[2] < 8,
            "a blacked-out audience must not retain a glowing wake, got {dark:?}"
        );
    }
}
