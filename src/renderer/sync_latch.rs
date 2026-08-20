//! The B14 sync-latch stage executor: the tape/NTSC horizontal shear on the
//! one slot-0 seam every audience path shares. Live LegacyExact, live
//! Advanced, the selective-VHS path, and export all converge on composite
//! slot 0 between the B8 melting edge and the B4 display stage, so this
//! single implementation and single shader serve all four — the
//! `encode_opaque_output` precedent, exactly the B8 and B4 shape.
//!
//! The stage owns **no texture at all**. Its entire state is the bounded
//! per-line offset table in [`crate::sync_latch::SyncLatchState`] — a few
//! kilobytes of host memory — plus the one uniform buffer that carries it to
//! the GPU. That is the whole point of the tranche: bounded state may latch,
//! but it may never grow, so latching costs one flag and one fixed table
//! rather than an accumulating buffer, and no resource law is threatened.
//!
//! The table is **program memory**, on the temporal-ring and bus-melt
//! precedent rather than B4's phosphor: blackout darkens the audience without
//! erasing the accumulated shear, and releasing the switch resumes from the
//! damage the cut interrupted. `reset_for` clears it on exactly the causes
//! that begin a new program — a patch generation, an Apply Look, a broad
//! revert — and deliberately not on a source cut or seek, which are moves
//! inside the same program.
//!
//! CPU reference: `sync_latch.rs`, followed expression for expression by
//! `sync_latch.wgsl`.

use crate::effects::params::TEMPORAL_REFERENCE_FPS;
use crate::sync_latch::{
    SyncLatchGpuHeader, SyncLatchParams, SyncLatchState, SYNC_LATCH_UNIFORM_BYTES,
};
use crate::temporal::TemporalResetCause;

pub struct SyncLatchGpu {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dimensions: [u32; 2],
    /// Built once against the renderer-lifetime composite views. Output size
    /// only ever changes by rebuilding the renderer, which rebuilds this
    /// stage with a cleared table.
    bind_group: Option<wgpu::BindGroup>,
    state: SyncLatchState,
    /// The stage's own 30 Hz reference clock — the `history_ticks_for_delta`
    /// law, owned here because the Exact temporal state does not advance on
    /// Advanced frames while this stage runs on every path.
    tick_accumulator: f64,
    total_ticks: u64,
}

impl SyncLatchGpu {
    pub fn new(
        device: &wgpu::Device,
        composite_format: wgpu::TextureFormat,
        dimensions: [u32; 2],
    ) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sync latch bind layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(SYNC_LATCH_UNIFORM_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sync latch shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sync_latch.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sync latch pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sync latch pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_latch"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_latch"),
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
        // Repeat on U is the tape wrap: a line that slides off one side
        // arrives at the other, and the bilinear tap that straddles the seam
        // filters across it rather than clamping. V never moves.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sync latch sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sync latch uniforms"),
            size: SYNC_LATCH_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_layout,
            sampler,
            uniform,
            dimensions,
            bind_group: None,
            state: SyncLatchState::new(dimensions[1]),
            tick_accumulator: 0.0,
            total_ticks: 0,
        }
    }

    /// Clear the accumulated table on the causes that begin a new program. A
    /// source cut or a seek keeps it: those move inside the same program, and
    /// the shears already suffered are part of its state.
    pub fn reset_for(&mut self, cause: TemporalResetCause) {
        let clears = matches!(
            cause,
            TemporalResetCause::PatchGeneration
                | TemporalResetCause::ApplyLook
                | TemporalResetCause::BroadRevert
                | TemporalResetCause::Resize
                | TemporalResetCause::ManualClear
        );
        if clears {
            self.state.clear();
        }
    }

    /// Whether the stage currently holds accumulated displacement. Published
    /// so the operator can see that the program is still broken.
    pub fn has_damage(&self) -> bool {
        self.state.has_damage()
    }

    fn ensure_bind_group(
        &mut self,
        device: &wgpu::Device,
        composite_views: &[wgpu::TextureView; 3],
    ) {
        if self.bind_group.is_some() {
            return;
        }
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sync latch group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(SYNC_LATCH_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }

    /// Encode the stage for one frame. Returns whether anything was encoded.
    ///
    /// A frame whose table displaces nothing — the authored default, an
    /// inactive stage, or simply a tick on which no band lost sync — encodes
    /// no pass at all, so slot 0 reaches the display stage byte for byte
    /// rather than being resampled through an identity.
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
        params: &SyncLatchParams,
        seed: u32,
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

        // The table is advanced unconditionally: this is the one call that
        // performs a release, and skipping it while the stage looked inert
        // would leave a released table still showing its last shear.
        self.state.advance(
            *params,
            self.total_ticks,
            u32::try_from(elapsed).unwrap_or(u32::MAX),
            seed,
        );

        if !self.state.applied().iter().any(|offset| *offset != 0.0) {
            return false;
        }

        self.ensure_bind_group(device, composite_views);
        let header = SyncLatchGpuHeader::from_parts(self.dimensions, self.state.line_count(), true);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&header));
        queue.write_buffer(
            &self.uniform,
            std::mem::size_of::<SyncLatchGpuHeader>() as u64,
            bytemuck::cast_slice(self.state.applied()),
        );

        // The shear reads slot 0, ends in slot 2, and returns to slot 0 — the
        // slot dance the melting edge and display stage already share.
        let group = self.bind_group.as_ref().expect("ensured above");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Sync latch pass"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        copy_full_frame(
            encoder,
            &composite_textures[2],
            &composite_textures[0],
            self.dimensions,
        );
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
    use crate::sync_latch::{sampled_u, SYNC_LATCH_SLIP_UV};

    const SIZE: u32 = 32;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Sync latch test"),
            ..Default::default()
        }))
        .ok()
    }

    fn composites(device: &wgpu::Device) -> ([wgpu::Texture; 3], [wgpu::TextureView; 3]) {
        let textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Sync latch fixture composite {i}")),
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

    /// A horizontal ramp: red carries the X coordinate, so a sheared line
    /// reports exactly how far it moved and in which direction.
    fn write_ramp(queue: &wgpu::Queue, texture: &wgpu::Texture) {
        let mut bytes = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for _y in 0..SIZE {
            for x in 0..SIZE {
                bytes.extend_from_slice(&[(x * 8) as u8, 0, 0, 255]);
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

    fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
        let bytes_per_row = (SIZE * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sync latch readback"),
            size: u64::from(bytes_per_row) * u64::from(SIZE),
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
                    bytes_per_row: Some(bytes_per_row),
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
        let mut rows = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for y in 0..SIZE {
            let start = (y * bytes_per_row) as usize;
            rows.extend_from_slice(&mapped[start..start + (SIZE * 4) as usize]);
        }
        drop(mapped);
        staging.unmap();
        rows
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn gpu_sync_latch_shears_each_line_by_its_own_offset_and_default_encodes_nothing() {
        let Some((device, queue)) = acquire_device() else {
            eprintln!("no adapter; skipping");
            return;
        };
        let (textures, views) = composites(&device);
        let mut stage = SyncLatchGpu::new(&device, FORMAT, [SIZE, SIZE]);

        // 1. The authored default encodes nothing at all, so slot 0 is
        //    untouched — byte-identical to the stage not existing.
        write_ramp(&queue, &textures[0]);
        let before = read_back(&device, &queue, &textures[0]);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let encoded = stage.encode(
            &device,
            &queue,
            &mut encoder,
            &textures,
            &views,
            &SyncLatchParams::default(),
            0,
            1.0 / 30.0,
        );
        queue.submit(std::iter::once(encoder.finish()));
        assert!(!encoded, "the authored default encoded a pass");
        assert_eq!(
            before,
            read_back(&device, &queue, &textures[0]),
            "a dormant stage moved a pixel"
        );

        // 2. An armed latched stage shears, and every line lands where the
        //    CPU reference says it should.
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut stage = SyncLatchGpu::new(&device, FORMAT, [SIZE, SIZE]);
        write_ramp(&queue, &textures[0]);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let encoded = stage.encode(
            &device,
            &queue,
            &mut encoder,
            &textures,
            &views,
            &params,
            7,
            1.0 / 30.0,
        );
        queue.submit(std::iter::once(encoder.finish()));
        assert!(encoded, "an armed stage encoded nothing");

        let offsets: Vec<f32> = stage.state.applied().to_vec();
        assert!(
            offsets.iter().any(|offset| *offset != 0.0),
            "the armed table drew no slip at all"
        );
        let sheared = read_back(&device, &queue, &textures[0]);
        // The expectation models the shader exactly: wrap the coordinate with
        // the CPU reference's own law, then reproduce the filtering sampler's
        // bilinear tap in texel space with Repeat on U. A tap that straddles
        // the seam legitimately blends the last column with the first — that
        // is the tape wrap filtering across the join rather than clamping, and
        // a nearest-neighbour expectation would wrongly call it a defect.
        let wrapped_bilinear = |uv_x: f32, offset: f32| -> f32 {
            let position = sampled_u(uv_x, offset) * SIZE as f32 - 0.5;
            let base = position.floor();
            let blend = position - base;
            let texel = |index: i32| -> f32 { (index.rem_euclid(SIZE as i32) as f32) * 8.0 };
            let low = texel(base as i32);
            let high = texel(base as i32 + 1);
            low * (1.0 - blend) + high * blend
        };
        for y in 0..SIZE {
            let offset = offsets[y as usize];
            for x in 0..SIZE {
                let uv_x = (x as f32 + 0.5) / SIZE as f32;
                let expected = wrapped_bilinear(uv_x, offset);
                let actual = f32::from(sheared[((y * SIZE + x) * 4) as usize]);
                assert!(
                    (expected - actual).abs() <= 3.0,
                    "line {y} column {x}: expected ~{expected}, read {actual} (offset {offset})"
                );
            }
        }
        // And the shear demonstrably moved the picture: at least one line's
        // sampled column differs from where it started.
        let unsheared = (0..SIZE).filter(|y| offsets[*y as usize] == 0.0).count();
        assert!(
            unsheared < SIZE as usize,
            "no line was sheared at all, so the parity check proved nothing"
        );

        // 3. Releasing the switch unwinds the accumulation: the very next
        //    frame carries at most one transient slip, never the accumulated
        //    displacement.
        for tick in 0..40 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            stage.encode(
                &device,
                &queue,
                &mut encoder,
                &textures,
                &views,
                &params,
                7,
                1.0 / 30.0,
            );
            queue.submit(std::iter::once(encoder.finish()));
            let _ = tick;
        }
        assert!(stage.has_damage(), "a latched run accumulated nothing");
        let released = SyncLatchParams {
            latched: false,
            ..params
        };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        stage.encode(
            &device,
            &queue,
            &mut encoder,
            &textures,
            &views,
            &released,
            7,
            1.0 / 30.0,
        );
        queue.submit(std::iter::once(encoder.finish()));
        assert!(!stage.has_damage(), "release left damage behind");
        assert!(
            stage
                .state
                .applied()
                .iter()
                .all(|offset| offset.abs() <= SYNC_LATCH_SLIP_UV + 1e-7),
            "a released line still carried accumulated displacement"
        );
    }

    #[test]
    #[ignore = "requires a physical GPU adapter"]
    fn gpu_sync_latch_reset_clears_only_the_causes_that_begin_a_new_program() {
        let Some((device, queue)) = acquire_device() else {
            eprintln!("no adapter; skipping");
            return;
        };
        let (textures, views) = composites(&device);
        let mut stage = SyncLatchGpu::new(&device, FORMAT, [SIZE, SIZE]);
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        write_ramp(&queue, &textures[0]);
        for _ in 0..30 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            stage.encode(
                &device,
                &queue,
                &mut encoder,
                &textures,
                &views,
                &params,
                3,
                1.0 / 30.0,
            );
            queue.submit(std::iter::once(encoder.finish()));
        }
        assert!(stage.has_damage());

        // A move inside the same program keeps the damage.
        stage.reset_for(TemporalResetCause::SourceCut);
        assert!(stage.has_damage(), "a source cut erased the shear history");
        stage.reset_for(TemporalResetCause::Seek);
        assert!(stage.has_damage(), "a seek erased the shear history");
        stage.reset_for(TemporalResetCause::BlackoutTransition);
        assert!(
            stage.has_damage(),
            "blackout erased the shear history — the table is program memory"
        );

        // A new program has no history of shears.
        stage.reset_for(TemporalResetCause::BroadRevert);
        assert!(!stage.has_damage(), "a broad revert kept the shear history");
    }
}
