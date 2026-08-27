//! The P4c planar-conversion GPU executor: the CPU oracle's production twin.
//!
//! This began as the measurement-only candidate the P4c stop receipt
//! prescribed, and the tracked
//! `docs/evidence/p4c-planar-gpu-candidate-receipt.json` is the measurement
//! that promoted it: 62.5% staging reduction, delivery p95 down roughly 70%,
//! upload p95 down 14–28% on the 720p/1080p two-source matrix, with CPU/GPU
//! equality within one 8-bit code on real decoded frames. Its production
//! consumer is the layer upload seam (`layers::mod`), which converts an
//! admitted planar frame into the layer's source texture through one pass;
//! the opt-in equality battery, seam candidate, and integrated total-frame
//! fixture remain as the executor's standing proof.
//!
//! The conversion law is not restated here. The GPU uniforms derive from the
//! exact [`CpuConversionContract`] the CPU oracle consumes — live frames
//! carry it as the payload's [`PlanarConversionRecipe`] — so admission,
//! matrix coefficients, range normalization, and chroma siting cannot drift
//! between the two paths: there is one derivation, consumed twice. The WGSL
//! follows `PlanarImagePayload::to_rgba8_cpu_reference` expression for
//! expression — integer plane codes through `textureLoad` on uint textures
//! (never a filtering sampler, so no hidden hardware filtering law enters),
//! the same explicit bilinear over the chroma plane, the same limited/full
//! normalization, and the same Kr/Kb reconstruction. The declared equality
//! tolerance is one 8-bit code value per channel: the oracle computes in
//! `f64` and rounds half away from zero, the GPU computes in `f32` and the
//! `Rgba8Unorm` store rounds to nearest, so exact half-codes may land one
//! apart and nothing else may. The conversion writes source-encoded bytes,
//! so the production target is the layer texture's **non-sRGB twin view**:
//! stored bytes equal what `write_texture` of packed values would store.
//!
//! The production WGSL lives under `src/shaders/` and is included verbatim,
//! so `build.rs` folds it into the same shader-bundle identity as every other
//! production shader. Its independent behavioral proof remains the CPU/GPU
//! equality battery.

use bytemuck::{Pod, Zeroable};

use super::planar::{
    chroma_sample_offset, CpuConversionContract, PlanarConversionError, PlanarConversionRecipe,
    PlanarImageLayout, PlanarImagePayload, PlanarPixelFormat, PlanarPlaneKind,
};
use super::{SourceColorDescriptor, SourceColorRange};

/// The complete production conversion pass. The fragment mirrors the CPU
/// oracle expression for expression; see the module documentation for the
/// tolerance argument. Keeping the source under `src/shaders/` makes its
/// bytes part of the build-generated production shader-bundle identity.
const PLANAR_CONVERT_WGSL: &str = include_str!("../shaders/planar_convert.wgsl");

/// Exactly 48 bytes, compile-time asserted like every other uniform in the
/// tree. Format codes: 0 `Yuv420p8`, 1 `Nv12`, 2 `P010Le`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PlanarConvertUniforms {
    pub size: [u32; 2],
    pub format: u32,
    pub bit_depth: u32,
    pub range_full: u32,
    pub _pad0: u32,
    pub chroma_offset: [f32; 2],
    pub kr: f32,
    pub kb: f32,
    pub _pad1: [f32; 2],
}

pub const PLANAR_CONVERT_UNIFORM_BYTES: u64 = 48;
const _: () =
    assert!(std::mem::size_of::<PlanarConvertUniforms>() == PLANAR_CONVERT_UNIFORM_BYTES as usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanarGpuError {
    Conversion(PlanarConversionError),
    MissingPlane(PlanarPlaneKind),
}

impl std::fmt::Display for PlanarGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversion(error) => write!(formatter, "planar GPU conversion contract: {error}"),
            Self::MissingPlane(kind) => {
                write!(formatter, "planar GPU payload plane {kind:?} missing")
            }
        }
    }
}

impl std::error::Error for PlanarGpuError {}

impl From<PlanarConversionError> for PlanarGpuError {
    fn from(error: PlanarConversionError) -> Self {
        Self::Conversion(error)
    }
}

/// Derive the GPU uniform record from the one shared conversion contract.
/// Every refusal the CPU oracle makes — unspecified metadata, PQ/HLG,
/// unsupported matrix, descriptor mismatch — is therefore this function's
/// refusal too, by construction rather than by a parallel table.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the descriptor-based derivation is the equality fixtures' door; production frames arrive with a recipe"
    )
)]
pub fn planar_convert_uniforms(
    layout: PlanarImageLayout,
    color: SourceColorDescriptor,
) -> Result<PlanarConvertUniforms, PlanarConversionError> {
    let contract = CpuConversionContract::from_descriptor(layout.format, color)?;
    let (offset_x, offset_y) = chroma_sample_offset(contract.chroma_location)?;
    Ok(PlanarConvertUniforms {
        size: [layout.width, layout.height],
        format: match layout.format {
            PlanarPixelFormat::Yuv420p8 => 0,
            PlanarPixelFormat::Nv12 => 1,
            PlanarPixelFormat::P010Le => 2,
        },
        bit_depth: u32::from(contract.bit_depth),
        range_full: u32::from(contract.range == SourceColorRange::Full),
        _pad0: 0,
        chroma_offset: [offset_x as f32, offset_y as f32],
        kr: contract.kr as f32,
        kb: contract.kb as f32,
        _pad1: [0.0, 0.0],
    })
}

/// The production uniform derivation: a live planar frame carries its recipe
/// with the pixels, so the upload seam never re-derives color from telemetry.
/// Production delivery admits `Yuv420p8` only; the format code is fixed.
pub fn planar_convert_uniforms_from_recipe(
    width: u32,
    height: u32,
    recipe: PlanarConversionRecipe,
) -> PlanarConvertUniforms {
    PlanarConvertUniforms {
        size: [width, height],
        format: 0,
        bit_depth: u32::from(recipe.bit_depth),
        range_full: u32::from(recipe.full_range),
        _pad0: 0,
        chroma_offset: recipe.chroma_offset,
        kr: recipe.kr,
        kb: recipe.kb,
        _pad1: [0.0, 0.0],
    }
}

/// The wgpu texture format each admitted plane uploads into. Integer formats
/// only: `textureLoad` returns the exact stored code, so the shader sees the
/// same integers the CPU oracle reads from the packed allocation.
pub const fn plane_texture_format(
    format: PlanarPixelFormat,
    kind: PlanarPlaneKind,
) -> wgpu::TextureFormat {
    match (format, kind) {
        (PlanarPixelFormat::Yuv420p8, _) => wgpu::TextureFormat::R8Uint,
        (PlanarPixelFormat::Nv12, PlanarPlaneKind::Y) => wgpu::TextureFormat::R8Uint,
        (PlanarPixelFormat::Nv12, _) => wgpu::TextureFormat::Rg8Uint,
        (PlanarPixelFormat::P010Le, PlanarPlaneKind::Y) => wgpu::TextureFormat::R16Uint,
        (PlanarPixelFormat::P010Le, _) => wgpu::TextureFormat::Rg16Uint,
    }
}

/// Plane textures for one admitted layout, reused across frames of the same
/// raster/format — the pooling shape a production integration would take.
pub struct PlanarPlaneTextures {
    layout: PlanarImageLayout,
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
}

pub struct PlanarGpuConverter {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// Bound in the third chroma slot for the two-plane formats; 1×1 and
    /// never read, because the shader's format branch cannot reach it.
    dummy_view: wgpu::TextureView,
}

impl PlanarGpuConverter {
    pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn new(device: &wgpu::Device) -> Self {
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar convert bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(PLANAR_CONVERT_UNIFORM_BYTES),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planar convert shader"),
            source: wgpu::ShaderSource::Wgsl(PLANAR_CONVERT_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planar convert pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planar convert pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_convert"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_convert"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: Self::TARGET_FORMAT,
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
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planar convert uniforms"),
            size: PLANAR_CONVERT_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Planar convert dummy plane"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            pipeline,
            bind_layout,
            uniform,
            dummy_view,
        }
    }

    /// One integer texture per plane of the admitted layout.
    pub fn plane_textures(
        &self,
        device: &wgpu::Device,
        layout: PlanarImageLayout,
    ) -> PlanarPlaneTextures {
        let mut textures = Vec::with_capacity(layout.plane_count());
        let mut views = Vec::with_capacity(layout.plane_count());
        for plane in layout.planes() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Planar convert plane"),
                size: wgpu::Extent3d {
                    width: plane.width,
                    height: plane.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: plane_texture_format(layout.format, plane.kind),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            views.push(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            textures.push(texture);
        }
        PlanarPlaneTextures {
            layout,
            textures,
            views,
        }
    }

    /// Upload every tightly packed plane of one payload. This is the
    /// staging-byte side of the candidate claim: for 8-bit 4:2:0 these
    /// writes move 1.5 bytes per pixel where the packed path moves 4.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the prototype-payload door remains the equality fixtures'; production uploads take the byte-blob door"
        )
    )]
    pub fn upload_planes(
        &self,
        queue: &wgpu::Queue,
        textures: &PlanarPlaneTextures,
        payload: &PlanarImagePayload,
    ) -> Result<(), PlanarGpuError> {
        for (index, plane) in textures.layout.planes().iter().enumerate() {
            let data = payload
                .plane(plane.kind)
                .ok_or(PlanarGpuError::MissingPlane(plane.kind))?;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &textures.textures[index],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plane.row_bytes as u32),
                    rows_per_image: Some(plane.height),
                },
                wgpu::Extent3d {
                    width: plane.width,
                    height: plane.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }

    /// The production upload: one tightly packed plane blob (the P4a planar
    /// payload's exact packing, which is the [`PlanarImageLayout`] packing)
    /// written plane by plane. Byte length is validated against the layout
    /// before any write.
    pub fn upload_plane_bytes(
        &self,
        queue: &wgpu::Queue,
        textures: &PlanarPlaneTextures,
        bytes: &[u8],
    ) -> Result<(), PlanarGpuError> {
        if bytes.len() != textures.layout.byte_len() {
            return Err(PlanarGpuError::Conversion(
                PlanarConversionError::DescriptorMismatch,
            ));
        }
        for (index, plane) in textures.layout.planes().iter().enumerate() {
            let end = plane.offset.saturating_add(plane.byte_len);
            let data = bytes
                .get(plane.offset..end)
                .ok_or(PlanarGpuError::MissingPlane(plane.kind))?;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &textures.textures[index],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plane.row_bytes as u32),
                    rows_per_image: Some(plane.height),
                },
                wgpu::Extent3d {
                    width: plane.width,
                    height: plane.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: PlanarConvertUniforms) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        textures: &PlanarPlaneTextures,
    ) -> wgpu::BindGroup {
        let chroma_b = textures.views.get(2).unwrap_or(&self.dummy_view);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planar convert group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(PLANAR_CONVERT_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&textures.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&textures.views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(chroma_b),
                },
            ],
        })
    }

    /// Encode the single conversion pass into `target_view`.
    pub fn encode_convert(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Planar convert pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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

    /// The whole equality path in one call: plane textures, uploads, one
    /// pass, one submit; returns the `Rgba8Unorm` target for readback.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the equality fixtures' complete conversion door")
    )]
    pub fn convert_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        payload: &PlanarImagePayload,
        color: SourceColorDescriptor,
    ) -> Result<wgpu::Texture, PlanarGpuError> {
        let layout = payload.layout();
        let uniforms = planar_convert_uniforms(layout, color)?;
        let textures = self.plane_textures(device, layout);
        self.upload_planes(queue, &textures, payload)?;
        self.write_uniforms(queue, uniforms);
        let bind_group = self.bind_group(device, &textures);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Planar convert target"),
            size: wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.encode_convert(&mut encoder, &bind_group, &target_view);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(target)
    }
}

/// Generate one H.264 yuv420p test clip whose color truth is completely
/// declared (tv/bt709/bt709/left), so the planar admission ladder admits it.
/// Shared by the equality/candidate fixtures here and the labeled export
/// case; the generic `-color_*` options reach the container while libx264's
/// VUI needs its own parameter surface — without `-x264-params` the decoder
/// honestly reports primaries/transfer unspecified and admission refuses.
#[cfg(test)]
pub(crate) fn write_declared_test_clip(
    path: &std::path::Path,
    width: u32,
    height: u32,
    duration_seconds: f64,
) -> Result<(), String> {
    let output = std::process::Command::new(crate::host_paths::ffmpeg())
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={width}x{height}:rate=30:duration={duration_seconds}"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-chroma_sample_location",
            "left",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-y",
        ])
        .arg(path)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::planar::{
        PlanarAllocationBudget, PlanarDeliveryDecision, PlanarDeliveryPolicy, PlanarPlaneInput,
        PlanarPlaneInputs,
    };
    use super::super::{
        BitDepth, ChromaLocation, DescriptorProvenance, DescriptorValue, MatrixCoefficients,
        PixelFamily, TransferCharacteristic,
    };
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    fn color(
        bits: u8,
        range: SourceColorRange,
        matrix: MatrixCoefficients,
        transfer: TransferCharacteristic,
        location: ChromaLocation,
    ) -> SourceColorDescriptor {
        let declared = DescriptorProvenance::CodecDeclared;
        SourceColorDescriptor {
            pixel_family: DescriptorValue::new(PixelFamily::Yuv, declared),
            bit_depth: DescriptorValue::new(BitDepth::Bits(bits), declared),
            range: DescriptorValue::new(range, declared),
            matrix: DescriptorValue::new(matrix, declared),
            transfer: DescriptorValue::new(transfer, declared),
            chroma_location: DescriptorValue::new(location, declared),
            ..Default::default()
        }
    }

    #[test]
    fn uniforms_are_48_bytes_and_answer_from_the_one_shared_contract() {
        assert_eq!(
            std::mem::size_of::<PlanarConvertUniforms>(),
            PLANAR_CONVERT_UNIFORM_BYTES as usize
        );

        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, 4, 2).unwrap();
        let uniforms = planar_convert_uniforms(
            layout,
            color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Left,
            ),
        )
        .unwrap();
        assert_eq!(uniforms.size, [4, 2]);
        assert_eq!(uniforms.format, 0);
        assert_eq!(uniforms.bit_depth, 8);
        assert_eq!(uniforms.range_full, 0);
        assert_eq!(uniforms.chroma_offset, [0.0, 0.5]);
        assert!((uniforms.kr - 0.2126).abs() < 1e-6);
        assert!((uniforms.kb - 0.0722).abs() < 1e-6);

        // Refusals are the CPU oracle's refusals, structurally: the same
        // contract derivation answers both paths.
        let mut unspecified_chroma = color(
            8,
            SourceColorRange::Limited,
            MatrixCoefficients::Bt709,
            TransferCharacteristic::Bt709,
            ChromaLocation::Unspecified,
        );
        unspecified_chroma.chroma_location = DescriptorValue::new(
            ChromaLocation::Unspecified,
            DescriptorProvenance::Unspecified,
        );
        assert_eq!(
            planar_convert_uniforms(layout, unspecified_chroma).unwrap_err(),
            PlanarConversionError::UnspecifiedChromaLocation
        );
        assert_eq!(
            planar_convert_uniforms(
                layout,
                color(
                    8,
                    SourceColorRange::Limited,
                    MatrixCoefficients::Bt709,
                    TransferCharacteristic::Pq,
                    ChromaLocation::Left,
                ),
            )
            .unwrap_err(),
            PlanarConversionError::HdrToneMapRequired(TransferCharacteristic::Pq)
        );
        assert_eq!(
            planar_convert_uniforms(
                layout,
                color(
                    10,
                    SourceColorRange::Limited,
                    MatrixCoefficients::Bt709,
                    TransferCharacteristic::Bt709,
                    ChromaLocation::Left,
                ),
            )
            .unwrap_err(),
            PlanarConversionError::DescriptorMismatch
        );

        let p010 = PlanarImageLayout::new(PlanarPixelFormat::P010Le, 4, 2).unwrap();
        let p010_uniforms = planar_convert_uniforms(
            p010,
            color(
                10,
                SourceColorRange::Full,
                MatrixCoefficients::Bt2020Ncl,
                TransferCharacteristic::Bt2020_10,
                ChromaLocation::TopLeft,
            ),
        )
        .unwrap();
        assert_eq!(p010_uniforms.format, 2);
        assert_eq!(p010_uniforms.bit_depth, 10);
        assert_eq!(p010_uniforms.range_full, 1);
        assert_eq!(p010_uniforms.chroma_offset, [0.0, 0.0]);
    }

    #[test]
    fn planar_staging_bytes_meet_the_audit_reduction_floor() {
        let hd_420 = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, 1920, 1080).unwrap();
        let hd_packed = 1920_usize * 1080 * 4;
        assert_eq!(hd_420.byte_len(), 3_110_400);
        assert_eq!(hd_packed, 8_294_400);
        let reduction = 100.0 - (hd_420.byte_len() as f64) * 100.0 / hd_packed as f64;
        assert!(reduction >= 50.0, "8-bit 4:2:0 reduction {reduction} < 50%");

        let hd_nv12 = PlanarImageLayout::new(PlanarPixelFormat::Nv12, 1920, 1080).unwrap();
        assert_eq!(hd_nv12.byte_len(), 3_110_400);

        // P010 is a fidelity case, not a bandwidth case: 25% before
        // alignment, exactly the stop receipt's arithmetic.
        let hd_p010 = PlanarImageLayout::new(PlanarPixelFormat::P010Le, 1920, 1080).unwrap();
        assert_eq!(hd_p010.byte_len(), 6_220_800);
    }

    /// The recipe-based production derivation must agree exactly with the
    /// descriptor-based fixture derivation — one contract, two doors.
    #[test]
    fn recipe_uniforms_agree_with_the_descriptor_derivation() {
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, 6, 4).unwrap();
        let descriptor = color(
            8,
            SourceColorRange::Limited,
            MatrixCoefficients::Bt709,
            TransferCharacteristic::Bt709,
            ChromaLocation::Left,
        );
        let from_descriptor = planar_convert_uniforms(layout, descriptor).unwrap();
        let recipe =
            crate::video::planar::planar_conversion_recipe(PlanarPixelFormat::Yuv420p8, descriptor)
                .unwrap();
        assert_eq!(
            planar_convert_uniforms_from_recipe(6, 4, recipe),
            from_descriptor
        );
    }

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Planar delivery candidate"),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue, info))
    }

    fn read_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let bytes_per_row = (width * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planar convert readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
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
        let mut rows = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            rows.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
        }
        drop(mapped);
        staging.unmap();
        rows
    }

    /// The declared tolerance: `f64`-round-half-away CPU versus
    /// `f32`-round-nearest GPU may differ by one code and never more; alpha
    /// is exact.
    fn assert_within_one_code(case: &str, gpu: &[u8], cpu: &[u8]) -> u8 {
        assert_eq!(gpu.len(), cpu.len(), "{case}: length mismatch");
        let mut max_delta = 0_u8;
        for (index, (gpu_byte, cpu_byte)) in gpu.iter().zip(cpu).enumerate() {
            if index % 4 == 3 {
                assert_eq!(*gpu_byte, 255, "{case}: alpha must be opaque at {index}");
                assert_eq!(*cpu_byte, 255, "{case}: oracle alpha must be opaque");
                continue;
            }
            let delta = gpu_byte.abs_diff(*cpu_byte);
            assert!(
                delta <= 1,
                "{case}: |GPU-CPU| = {delta} at byte {index} (gpu {gpu_byte}, cpu {cpu_byte})"
            );
            max_delta = max_delta.max(delta);
        }
        max_delta
    }

    fn yuv420_payload(
        width: u32,
        height: u32,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        budget: &Arc<PlanarAllocationBudget>,
    ) -> PlanarImagePayload {
        let chroma_row = (width as usize).div_ceil(2);
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, width, height).unwrap();
        PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Yuv420p8 {
                y: PlanarPlaneInput::new(y, width as usize),
                u: PlanarPlaneInput::new(u, chroma_row),
                v: PlanarPlaneInput::new(v, chroma_row),
            },
            budget,
        )
        .unwrap()
    }

    fn p010_words(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| (value << 6).to_le_bytes())
            .collect()
    }

    /// Deterministic byte stream so the noise cases replay identically on
    /// every host; no RNG dependency enters the fixture.
    fn splitmix_bytes(seed: u64, count: usize) -> Vec<u8> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }

    fn convert_and_compare(
        case: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        converter: &PlanarGpuConverter,
        payload: &PlanarImagePayload,
        descriptor: SourceColorDescriptor,
    ) -> (Vec<u8>, u8) {
        let cpu = payload
            .to_rgba8_cpu_reference(descriptor)
            .unwrap_or_else(|error| panic!("{case}: CPU oracle refused: {error}"));
        let target = converter
            .convert_to_texture(device, queue, payload, descriptor)
            .unwrap_or_else(|error| panic!("{case}: GPU conversion refused: {error}"));
        let layout = payload.layout();
        let gpu = read_rgba8(device, queue, &target, layout.width, layout.height);
        let max_delta = assert_within_one_code(case, &gpu, &cpu.rgba);
        (gpu, max_delta)
    }

    /// The reopened P4c equality battery, synthetic half: every admitted
    /// format, the 601/709/2020 matrices, limited and full range, all six
    /// declared chroma sitings, odd dimensions, and deterministic noise —
    /// each within one 8-bit code of the CPU oracle.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_planar_conversion_matches_the_cpu_reference_battery() {
        let Some((device, queue, _info)) = acquire_device() else {
            eprintln!("no adapter; skipping");
            return;
        };
        let converter = PlanarGpuConverter::new(&device);
        let budget = PlanarAllocationBudget::new(64 * 1024 * 1024).unwrap();

        // BT.601 limited saturated landmarks.
        let bars = yuv420_payload(2, 2, &[16, 235, 81, 145], &[128], &[128], &budget);
        convert_and_compare(
            "bt601 limited neutral bars",
            &device,
            &queue,
            &converter,
            &bars,
            color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Smpte170M,
                TransferCharacteristic::Smpte170M,
                ChromaLocation::Left,
            ),
        );
        let red_bar = yuv420_payload(2, 2, &[81; 4], &[90], &[240], &budget);
        for (label, matrix, transfer) in [
            (
                "red bar bt601 limited",
                MatrixCoefficients::Smpte170M,
                TransferCharacteristic::Smpte170M,
            ),
            (
                "red bar bt709 limited",
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
            ),
        ] {
            convert_and_compare(
                label,
                &device,
                &queue,
                &converter,
                &red_bar,
                color(
                    8,
                    SourceColorRange::Limited,
                    matrix,
                    transfer,
                    ChromaLocation::Left,
                ),
            );
        }

        // BT.709 full-range ramp.
        let full = yuv420_payload(2, 2, &[0, 255, 128, 64], &[128], &[128], &budget);
        convert_and_compare(
            "bt709 full ramp",
            &device,
            &queue,
            &converter,
            &full,
            color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Center,
            ),
        );

        // All six declared sitings over a hard chroma edge; the sitings must
        // agree with the oracle individually and remain distinguishable.
        let chroma_edge = yuv420_payload(4, 2, &[128; 8], &[16, 240], &[240, 16], &budget);
        let mut siting_images = Vec::new();
        for location in [
            ChromaLocation::Left,
            ChromaLocation::Center,
            ChromaLocation::TopLeft,
            ChromaLocation::Top,
            ChromaLocation::BottomLeft,
            ChromaLocation::Bottom,
        ] {
            let (gpu, _) = convert_and_compare(
                &format!("chroma siting {location:?}"),
                &device,
                &queue,
                &converter,
                &chroma_edge,
                color(
                    8,
                    SourceColorRange::Full,
                    MatrixCoefficients::Bt709,
                    TransferCharacteristic::Bt709,
                    location,
                ),
            );
            siting_images.push(gpu);
        }
        assert_ne!(
            siting_images[0], siting_images[1],
            "left and center siting must reconstruct the edge differently"
        );

        // NV12: interleaved chroma reads the same pairs the oracle reads.
        let nv12_layout = PlanarImageLayout::new(PlanarPixelFormat::Nv12, 4, 2).unwrap();
        let nv12 = PlanarImagePayload::from_planes(
            nv12_layout,
            PlanarPlaneInputs::Nv12 {
                y: PlanarPlaneInput::new(&[16, 32, 64, 128, 235, 200, 100, 50], 4),
                uv: PlanarPlaneInput::new(&[128, 128, 90, 240], 4),
            },
            &budget,
        )
        .unwrap();
        convert_and_compare(
            "nv12 bt709 limited",
            &device,
            &queue,
            &converter,
            &nv12,
            color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Left,
            ),
        );

        // P010 BT.2020 limited: the 10-bit ramp and a colored pair.
        let p010_layout = PlanarImageLayout::new(PlanarPixelFormat::P010Le, 4, 2).unwrap();
        let p010 = PlanarImagePayload::from_planes(
            p010_layout,
            PlanarPlaneInputs::P010Le {
                y: PlanarPlaneInput::new(&p010_words(&[64, 256, 512, 940, 64, 256, 512, 940]), 8),
                uv: PlanarPlaneInput::new(&p010_words(&[600, 700, 512, 512]), 8),
            },
            &budget,
        )
        .unwrap();
        convert_and_compare(
            "p010 bt2020 limited",
            &device,
            &queue,
            &converter,
            &p010,
            color(
                10,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt2020Ncl,
                TransferCharacteristic::Bt2020_10,
                ChromaLocation::TopLeft,
            ),
        );

        // Odd dimensions exercise the ceil-half chroma geometry.
        let odd_y = splitmix_bytes(11, 15);
        let odd_u = splitmix_bytes(12, 6);
        let odd_v = splitmix_bytes(13, 6);
        let odd = yuv420_payload(5, 3, &odd_y, &odd_u, &odd_v, &budget);
        convert_and_compare(
            "odd 5x3 noise",
            &device,
            &queue,
            &converter,
            &odd,
            color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Center,
            ),
        );

        // Deterministic 64×36 noise per format.
        let noise_y = splitmix_bytes(21, 64 * 36);
        let noise_u = splitmix_bytes(22, 32 * 18);
        let noise_v = splitmix_bytes(23, 32 * 18);
        let noise_420 = yuv420_payload(64, 36, &noise_y, &noise_u, &noise_v, &budget);
        convert_and_compare(
            "noise yuv420p bt709 full",
            &device,
            &queue,
            &converter,
            &noise_420,
            color(
                8,
                SourceColorRange::Full,
                MatrixCoefficients::Bt709,
                TransferCharacteristic::Bt709,
                ChromaLocation::Center,
            ),
        );
        let noise_uv = splitmix_bytes(24, 32 * 18 * 2);
        let noise_nv12 = PlanarImagePayload::from_planes(
            PlanarImageLayout::new(PlanarPixelFormat::Nv12, 64, 36).unwrap(),
            PlanarPlaneInputs::Nv12 {
                y: PlanarPlaneInput::new(&noise_y, 64),
                uv: PlanarPlaneInput::new(&noise_uv, 64),
            },
            &budget,
        )
        .unwrap();
        convert_and_compare(
            "noise nv12 bt601 limited",
            &device,
            &queue,
            &converter,
            &noise_nv12,
            color(
                8,
                SourceColorRange::Limited,
                MatrixCoefficients::Smpte170M,
                TransferCharacteristic::Smpte170M,
                ChromaLocation::Left,
            ),
        );
        let noise_y10: Vec<u16> = splitmix_bytes(25, 64 * 36)
            .iter()
            .zip(splitmix_bytes(26, 64 * 36))
            .map(|(high, low)| ((u16::from(*high) << 2) | u16::from(low & 3)).min(1023))
            .collect();
        let noise_uv10: Vec<u16> = splitmix_bytes(27, 32 * 18 * 2)
            .iter()
            .zip(splitmix_bytes(28, 32 * 18 * 2))
            .map(|(high, low)| ((u16::from(*high) << 2) | u16::from(low & 3)).min(1023))
            .collect();
        let noise_p010 = PlanarImagePayload::from_planes(
            PlanarImageLayout::new(PlanarPixelFormat::P010Le, 64, 36).unwrap(),
            PlanarPlaneInputs::P010Le {
                y: PlanarPlaneInput::new(&p010_words(&noise_y10), 128),
                uv: PlanarPlaneInput::new(&p010_words(&noise_uv10), 128),
            },
            &budget,
        )
        .unwrap();
        convert_and_compare(
            "noise p010 bt2020 limited",
            &device,
            &queue,
            &converter,
            &noise_p010,
            color(
                10,
                SourceColorRange::Limited,
                MatrixCoefficients::Bt2020Ncl,
                TransferCharacteristic::Bt2020_10,
                ChromaLocation::TopLeft,
            ),
        );
    }

    struct CandidateMedia {
        root: std::path::PathBuf,
        sources: Vec<(String, std::path::PathBuf, u32, u32)>,
    }

    impl CandidateMedia {
        fn generate() -> Result<Self, String> {
            let root = std::env::temp_dir().join(format!(
                "collideoscope-p4c-candidate-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            ));
            std::fs::create_dir(&root).map_err(|error| error.to_string())?;
            let mut sources = Vec::new();
            for (label, width, height) in [("720p", 1280_u32, 720_u32), ("1080p", 1920, 1080)] {
                let path = root.join(format!("candidate-{label}.mp4"));
                if let Err(error) = super::write_declared_test_clip(&path, width, height, 8.5) {
                    let _ = std::fs::remove_dir_all(&root);
                    return Err(error);
                }
                sources.push((label.to_owned(), path, width, height));
            }
            Ok(Self { root, sources })
        }
    }

    impl Drop for CandidateMedia {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn percentile_ns(sorted: &[u64], percent: usize) -> u64 {
        let rank = (sorted.len() * percent).div_ceil(100).max(1);
        sorted[rank - 1]
    }

    struct SourceMeasurement {
        width: u32,
        height: u32,
        measured_frames: usize,
        packed_delivery_ns: Vec<u64>,
        planar_delivery_ns: Vec<u64>,
        spot_payloads: Vec<(usize, PlanarImagePayload)>,
        first_packed_rgba: Vec<u8>,
        descriptor: SourceColorDescriptor,
    }

    /// Decode one fixture source with the production-shaped software loop
    /// and measure, per selected frame, the packed delivery law (swscale to
    /// RGBA plus the stride-aware repack) against the planar delivery law
    /// (row-copying the decoder's planes into one immutable planar
    /// allocation). The decode itself is common to both paths and excluded.
    fn measure_delivery(
        label: &str,
        path: &std::path::Path,
        frames_to_measure: usize,
    ) -> SourceMeasurement {
        use ffmpeg_next as ffmpeg;
        use std::time::Instant;

        let path_text = path.to_string_lossy().into_owned();

        // The production descriptor freeze: open the real decoder once and
        // take its frozen color truth; the fixture never invents metadata.
        let mut production = crate::video::VideoDecoder::open(&path_text).expect("decoder open");
        production
            .next_timed_frame_result(1)
            .expect("first production frame");
        let descriptor = production.source_color_descriptor();
        let field_order = production.source_display_descriptor().field_order.value;
        drop(production);
        assert_eq!(
            super::super::planar::planar_delivery_decision(
                PlanarDeliveryPolicy::MetadataManaged,
                PlanarPixelFormat::Yuv420p8,
                descriptor,
                field_order,
            ),
            PlanarDeliveryDecision::PrototypePlanar(PlanarPixelFormat::Yuv420p8),
            "{label}: the fixture source must pass the real admission law \
             (declared range/matrix/transfer/chroma siting, progressive)"
        );

        let mut input = super::super::decoder::open_input(
            &path_text,
            Arc::new(AtomicBool::new(false)),
            "p4c planar candidate",
        )
        .expect("candidate open");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("video stream");
        let stream_index = stream.index();
        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .expect("codec context");
        let mut decoder = context.decoder().video().expect("software decoder");

        let budget = PlanarAllocationBudget::new(64 * 1024 * 1024).unwrap();
        let spot_ordinals = [0_usize, frames_to_measure / 2, frames_to_measure - 1];

        let mut measurement = SourceMeasurement {
            width: 0,
            height: 0,
            measured_frames: 0,
            packed_delivery_ns: Vec::with_capacity(frames_to_measure),
            planar_delivery_ns: Vec::with_capacity(frames_to_measure),
            spot_payloads: Vec::new(),
            first_packed_rgba: Vec::new(),
            descriptor,
        };
        let mut scaler: Option<ffmpeg::software::scaling::Context> = None;
        let mut rgba_frame = ffmpeg::util::frame::video::Video::empty();
        let mut layout: Option<PlanarImageLayout> = None;

        let mut process = |frame: &ffmpeg::util::frame::video::Video,
                           measurement: &mut SourceMeasurement|
         -> bool {
            if measurement.measured_frames >= frames_to_measure {
                return false;
            }
            let width = frame.width();
            let height = frame.height();
            if measurement.measured_frames == 0 {
                assert_eq!(
                    frame.format(),
                    ffmpeg::format::Pixel::YUV420P,
                    "{label}: candidate decode must produce yuv420p"
                );
                measurement.width = width;
                measurement.height = height;
                let mut context = ffmpeg::software::scaling::Context::get(
                    frame.format(),
                    width,
                    height,
                    ffmpeg::format::Pixel::RGBA,
                    width,
                    height,
                    ffmpeg::software::scaling::flag::Flags::BILINEAR,
                )
                .expect("scaler");
                crate::video::source_descriptor::configure_sws_conversion(
                    &mut context,
                    measurement.descriptor,
                );
                scaler = Some(context);
                layout = Some(
                    PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, width, height)
                        .expect("planar layout"),
                );
            }

            // Packed delivery: the production law's shape — one swscale run
            // and one stride-aware row repack into a fresh allocation.
            let packed_start = Instant::now();
            scaler
                .as_mut()
                .expect("scaler ready")
                .run(frame, &mut rgba_frame)
                .expect("swscale run");
            let row_bytes = width as usize * 4;
            let stride = rgba_frame.stride(0);
            let data = rgba_frame.data(0);
            let mut packed = Vec::with_capacity(row_bytes * height as usize);
            for row in 0..height as usize {
                let start = row * stride;
                packed.extend_from_slice(&data[start..start + row_bytes]);
            }
            let packed_ns = packed_start.elapsed().as_nanos() as u64;

            // Planar delivery: row-copying the decoder's own planes into the
            // immutable planar allocation; no swscale at all.
            let planar_start = Instant::now();
            let payload = PlanarImagePayload::from_planes(
                layout.expect("layout ready"),
                PlanarPlaneInputs::Yuv420p8 {
                    y: PlanarPlaneInput::new(frame.data(0), frame.stride(0)),
                    u: PlanarPlaneInput::new(frame.data(1), frame.stride(1)),
                    v: PlanarPlaneInput::new(frame.data(2), frame.stride(2)),
                },
                &budget,
            )
            .expect("planar payload");
            let planar_ns = planar_start.elapsed().as_nanos() as u64;

            measurement.packed_delivery_ns.push(packed_ns);
            measurement.planar_delivery_ns.push(planar_ns);
            if spot_ordinals.contains(&measurement.measured_frames) {
                measurement
                    .spot_payloads
                    .push((measurement.measured_frames, payload));
            }
            if measurement.measured_frames == 0 {
                measurement.first_packed_rgba = packed;
            }
            measurement.measured_frames += 1;
            true
        };

        'decode: for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet).expect("send packet");
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if !process(&decoded, &mut measurement) {
                    break 'decode;
                }
            }
        }
        if measurement.measured_frames < frames_to_measure {
            decoder.send_eof().ok();
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if !process(&decoded, &mut measurement) {
                    break;
                }
            }
        }
        assert_eq!(
            measurement.measured_frames, frames_to_measure,
            "{label}: fixture source decoded too few frames"
        );
        measurement
    }

    /// The Phase B live-seam proof: a real `Layer` over a real threaded
    /// decoder, driven through the production harvest/upload path.
    ///
    /// Three claims in one fixture. Legacy first: a packed frame uploaded
    /// through the real seam stores exactly its payload bytes — the legacy
    /// byte law at the integrated seam, not merely at the decoder. Managed
    /// next: after `set_delivery_policy(MetadataManaged)` the decoder
    /// delivers a planar frame whose recipe rides the payload, the upload
    /// converts it in place, the layer publishes `delivery_active_planar`,
    /// and the texture agrees with the CPU oracle within one code. Legacy
    /// again: flipping back mid-session restores the exact packed store, so
    /// the policy is live in both directions.
    #[test]
    #[ignore = "requires a GPU adapter and an ffmpeg executable; proves the live planar upload seam"]
    fn gpu_planar_integration_layer_upload_matches_the_cpu_oracle_and_legacy_stays_exact() {
        use crate::layers::LayerSource;
        use crate::video::planar::PlanarDeliveryPolicy;

        ffmpeg_next::init().ok();
        let Some((device, queue, _info)) = acquire_device() else {
            panic!("the live-seam fixture requires a GPU adapter");
        };
        let root = std::env::temp_dir().join(format!(
            "collideoscope-p4c-live-seam-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).expect("fixture root");
        let clip = root.join("live-seam.mp4");
        super::write_declared_test_clip(&clip, 128, 72, 4.0).expect("declared clip");
        let clip_text = clip.to_string_lossy().into_owned();

        // The frozen declared truth for the oracle, from the production
        // sync decoder.
        let mut sync = crate::video::VideoDecoder::open(&clip_text).expect("sync open");
        sync.next_timed_frame_result(1).expect("sync first frame");
        let descriptor = sync.source_color_descriptor();
        drop(sync);

        let mut layer = crate::layers::Layer::new_with_media_policy(
            &clip_text,
            &device,
            &crate::media_safety::MediaSafetyPolicy::safe(),
        )
        .expect("layer open");

        let pump = |layer: &mut crate::layers::Layer| -> crate::video::threaded::ReadyFrame {
            let started = std::time::Instant::now();
            loop {
                if let Some(frame) = layer.take_ready_media_frame().expect("harvest") {
                    return frame;
                }
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "no frame arrived within the harvest deadline"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        let request = |layer: &mut crate::layers::Layer, seconds: f64| {
            let LayerSource::Video(decoder) = &mut layer.source else {
                panic!("fixture layer must be video");
            };
            decoder.request_seek(seconds).expect("seek request");
        };
        // The production loop drains completed upload validations once per
        // event-loop turn after a device poll; the fixture does the same so
        // the bounded pending queue never fills between uploads.
        let drain_validation = |layer: &mut crate::layers::Layer| {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU wait");
            for _ in 0..4 {
                assert!(
                    layer.poll_upload_validation(1).is_none(),
                    "an upload raised a GPU validation fault"
                );
            }
        };

        // 1. Legacy byte law at the integrated seam.
        let mut seed = pump(&mut layer);
        assert_eq!(
            seed.rgba.layout().format,
            crate::video::DecodedPixelFormat::PackedRgba8,
            "the open seed decodes under the legacy default"
        );
        layer
            .upload_ready_media_frame(&device, &queue, 1, &mut seed)
            .expect("legacy upload");
        drain_validation(&mut layer);
        assert!(!layer.delivery_active_planar());
        let stored = read_rgba8(&device, &queue, &layer.texture, 128, 72);
        assert_eq!(
            stored,
            seed.rgba.as_slice(),
            "a packed upload must store exactly its payload bytes"
        );

        // 2. Managed planar delivery through the same seam.
        layer.set_delivery_policy(PlanarDeliveryPolicy::MetadataManaged);
        request(&mut layer, 1.0);
        let mut planar_frame = pump(&mut layer);
        for _ in 0..100 {
            if planar_frame.rgba.layout().format == crate::video::DecodedPixelFormat::PlanarYuv420p8
            {
                break;
            }
            // An in-flight packed completion may land first; keep pumping.
            request(&mut layer, 1.5);
            planar_frame = pump(&mut layer);
        }
        assert_eq!(
            planar_frame.rgba.layout().format,
            crate::video::DecodedPixelFormat::PlanarYuv420p8,
            "the managed policy must deliver a planar frame"
        );
        assert!(planar_frame.rgba.conversion_recipe().is_some());
        layer
            .upload_ready_media_frame(&device, &queue, 2, &mut planar_frame)
            .expect("planar upload");
        drain_validation(&mut layer);
        assert!(layer.delivery_active_planar());

        // Oracle agreement on the exact delivered planes.
        let blob = planar_frame.rgba.as_slice();
        let budget = PlanarAllocationBudget::new(4 * 1024 * 1024).unwrap();
        let layout = PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, 128, 72).unwrap();
        let luma_len = 128 * 72;
        let chroma_len = 64 * 36;
        let oracle_payload = PlanarImagePayload::from_planes(
            layout,
            PlanarPlaneInputs::Yuv420p8 {
                y: PlanarPlaneInput::new(&blob[..luma_len], 128),
                u: PlanarPlaneInput::new(&blob[luma_len..luma_len + chroma_len], 64),
                v: PlanarPlaneInput::new(&blob[luma_len + chroma_len..], 64),
            },
            &budget,
        )
        .unwrap();
        let cpu = oracle_payload
            .to_rgba8_cpu_reference(descriptor)
            .expect("CPU oracle");
        let gpu = read_rgba8(&device, &queue, &layer.texture, 128, 72);
        assert_within_one_code("live planar upload", &gpu, &cpu.rgba);

        // 3. Flipping back restores the exact packed store.
        layer.set_delivery_policy(PlanarDeliveryPolicy::LegacyRgba);
        request(&mut layer, 2.0);
        let mut packed_again = pump(&mut layer);
        for _ in 0..100 {
            if packed_again.rgba.layout().format == crate::video::DecodedPixelFormat::PackedRgba8 {
                break;
            }
            request(&mut layer, 2.5);
            packed_again = pump(&mut layer);
        }
        assert_eq!(
            packed_again.rgba.layout().format,
            crate::video::DecodedPixelFormat::PackedRgba8
        );
        layer
            .upload_ready_media_frame(&device, &queue, 3, &mut packed_again)
            .expect("legacy upload after flip");
        drain_validation(&mut layer);
        assert!(!layer.delivery_active_planar());
        let stored = read_rgba8(&device, &queue, &layer.texture, 128, 72);
        assert_eq!(stored, packed_again.rgba.as_slice());

        drop(layer);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The reopened P4c seam measurement: the audit's 720p/1080p two-source
    /// CPU/GPU equality plus delivery/upload p95/p99 fixture. The promoted
    /// Phase-A receipt under `docs/evidence/` is immutable historical
    /// evidence; reruns write a distinct ignored follow-up under `target/`.
    /// Performance numbers are recorded, never asserted — a truthful
    /// negative result is a valid completion, exactly as it was for D3D11VA.
    #[test]
    #[ignore = "requires a GPU adapter and an ffmpeg executable; emits an untracked P4c seam follow-up receipt"]
    fn gpu_planar_delivery_followup_measures_conversion_upload_and_writes_untracked_receipt() {
        use std::time::Instant;

        if cfg!(debug_assertions) {
            panic!("timing evidence must come from --release; debug timings are not evidence");
        }
        ffmpeg_next::init().ok();

        let Some((device, queue, adapter)) = acquire_device() else {
            panic!("the seam follow-up receipt requires a GPU adapter");
        };
        let converter = PlanarGpuConverter::new(&device);
        let media = CandidateMedia::generate().expect("generate follow-up sources");

        const MEASURED_FRAMES: usize = 240;
        const UPLOAD_WARMUP: usize = 60;
        const UPLOAD_ITERATIONS: usize = 240;

        let mut source_reports = Vec::new();
        for (label, path, width, height) in &media.sources {
            let mut measurement = measure_delivery(label, path, MEASURED_FRAMES);
            assert_eq!(measurement.width, *width);
            assert_eq!(measurement.height, *height);

            // CPU/GPU equality on real decoded frames at three ordinals.
            let mut equality_max_delta = 0_u8;
            for (ordinal, payload) in &measurement.spot_payloads {
                let (_, delta) = convert_and_compare(
                    &format!("{label} frame {ordinal}"),
                    &device,
                    &queue,
                    &converter,
                    payload,
                    measurement.descriptor,
                );
                equality_max_delta = equality_max_delta.max(delta);
            }
            assert_eq!(
                measurement.spot_payloads.len(),
                3,
                "{label}: three equality spot frames expected"
            );

            // Upload benchmark: the same real first frame through both laws.
            let layout =
                PlanarImageLayout::new(PlanarPixelFormat::Yuv420p8, *width, *height).unwrap();
            let uniforms = planar_convert_uniforms(layout, measurement.descriptor)
                .expect("candidate uniforms");
            let textures = converter.plane_textures(&device, layout);
            converter.write_uniforms(&queue, uniforms);
            let bind_group = converter.bind_group(&device, &textures);
            let packed_target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Packed upload target"),
                size: wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PlanarGpuConverter::TARGET_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let planar_target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Planar convert upload target"),
                size: wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PlanarGpuConverter::TARGET_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let planar_target_view =
                planar_target.create_view(&wgpu::TextureViewDescriptor::default());
            let packed_bytes = std::mem::take(&mut measurement.first_packed_rgba);
            let first_payload = &measurement
                .spot_payloads
                .iter()
                .find(|(ordinal, _)| *ordinal == 0)
                .expect("frame 0 payload")
                .1;
            assert_eq!(
                packed_bytes.len(),
                (*width as usize) * (*height as usize) * 4
            );

            let fence = || {
                queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU wait");
            };
            let packed_iteration = || {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &packed_target,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &packed_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(*height),
                    },
                    wgpu::Extent3d {
                        width: *width,
                        height: *height,
                        depth_or_array_layers: 1,
                    },
                );
                fence();
            };
            let planar_iteration = || {
                converter
                    .upload_planes(&queue, &textures, first_payload)
                    .expect("plane upload");
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                converter.encode_convert(&mut encoder, &bind_group, &planar_target_view);
                queue.submit(std::iter::once(encoder.finish()));
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU wait");
            };
            for _ in 0..UPLOAD_WARMUP {
                packed_iteration();
                planar_iteration();
            }
            let mut packed_upload_ns = Vec::with_capacity(UPLOAD_ITERATIONS);
            for _ in 0..UPLOAD_ITERATIONS {
                let start = Instant::now();
                packed_iteration();
                packed_upload_ns.push(start.elapsed().as_nanos() as u64);
            }
            let mut planar_upload_ns = Vec::with_capacity(UPLOAD_ITERATIONS);
            for _ in 0..UPLOAD_ITERATIONS {
                let start = Instant::now();
                planar_iteration();
                planar_upload_ns.push(start.elapsed().as_nanos() as u64);
            }

            measurement.packed_delivery_ns.sort_unstable();
            measurement.planar_delivery_ns.sort_unstable();
            packed_upload_ns.sort_unstable();
            planar_upload_ns.sort_unstable();

            let packed_frame_bytes = (*width as u64) * (*height as u64) * 4;
            let planar_frame_bytes = layout.byte_len() as u64;
            let staging_reduction_percent =
                100.0 - (planar_frame_bytes as f64) * 100.0 / packed_frame_bytes as f64;
            assert!(
                staging_reduction_percent >= 50.0,
                "{label}: the audit's 50% staging floor must hold for 8-bit 4:2:0"
            );

            let stats = |sorted: &[u64]| {
                serde_json::json!({
                    "p50_ns": percentile_ns(sorted, 50),
                    "p95_ns": percentile_ns(sorted, 95),
                    "p99_ns": percentile_ns(sorted, 99),
                })
            };
            let improvement = |packed: &[u64], planar: &[u64]| -> f64 {
                let packed_p95 = percentile_ns(packed, 95) as f64;
                (packed_p95 - percentile_ns(planar, 95) as f64) * 100.0 / packed_p95
            };
            source_reports.push(serde_json::json!({
                "label": label,
                "width": width,
                "height": height,
                "measured_frames": measurement.measured_frames,
                "admission": "PrototypePlanar(Yuv420p8), the compatibility variant returned by planar_delivery_decision on the production-frozen descriptor",
                "equality": {
                    "spot_frames": [0, MEASURED_FRAMES / 2, MEASURED_FRAMES - 1],
                    "tolerance": "one 8-bit code value per channel; alpha exact",
                    "max_channel_delta": equality_max_delta,
                },
                "bytes_per_frame": {
                    "packed_rgba8": packed_frame_bytes,
                    "planar_yuv420p8": planar_frame_bytes,
                    "staging_reduction_percent": staging_reduction_percent,
                },
                "delivery_ns": {
                    "law": "packed = one swscale run plus stride-aware repack into a fresh allocation; planar = row-copy of the decoder's planes into one immutable planar allocation; decode excluded from both",
                    "packed": stats(&measurement.packed_delivery_ns),
                    "planar": stats(&measurement.planar_delivery_ns),
                    "planar_p95_improvement_percent": improvement(
                        &measurement.packed_delivery_ns,
                        &measurement.planar_delivery_ns,
                    ),
                },
                "upload_ns": {
                    "law": format!(
                        "packed = write_texture of {packed_frame_bytes} RGBA bytes, fenced; planar = write_texture of {planar_frame_bytes} plane bytes plus one conversion pass, fenced; {UPLOAD_WARMUP} warm-up and {UPLOAD_ITERATIONS} measured iterations each"
                    ),
                    "packed": stats(&packed_upload_ns),
                    "planar": stats(&planar_upload_ns),
                    "planar_p95_improvement_percent": improvement(
                        &packed_upload_ns,
                        &planar_upload_ns,
                    ),
                },
            }));
        }

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let receipt = serde_json::json!({
            "schema": "collide-o-scope-p4c-planar-gpu-followup-receipt/1",
            "command": "cargo test --locked --release --bin collide-o-scope gpu_planar_delivery_followup_measures_conversion_upload_and_writes_untracked_receipt -- --ignored --nocapture",
            "measured_at": {
                "commit": git(&["rev-parse", "HEAD"]),
                "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
                "tree": match git(&["status", "--porcelain"]).as_str() {
                    "unknown" => "unknown",
                    "" => "clean",
                    _ => "dirty",
                },
            },
            "host": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "build_profile": "release",
            },
            "adapter": {
                "name": adapter.name,
                "backend": format!("{:?}", adapter.backend),
                "driver": adapter.driver,
                "driver_info": adapter.driver_info,
            },
            "scope": "production-seam follow-up only: authored metadata_managed delivery is integrated in live and export, but this fixture isolates decoder materialization and upload. It does not measure integrated total-frame latency, does not change legacy_rgba as the default, and makes no auto-selection claim",
            "sources": "two generated H.264 yuv420p clips (testsrc2, 30 fps) with declared tv/bt709/bt709/left metadata; the real decoder's frozen descriptor drives admission and conversion",
            "conversion_law": "source_encoded_sdr_rgba8_no_gamut_mapping — the CPU oracle's law; the GPU twin derives uniforms from the same CpuConversionContract",
            "sources_measured": source_reports,
        });
        std::fs::create_dir_all("target").expect("create ignored receipt directory");
        std::fs::write(
            "target/p4c-planar-gpu-followup-receipt.json",
            format!("{}\n", serde_json::to_string_pretty(&receipt).unwrap()),
        )
        .expect("write the untracked P4c follow-up receipt");
        println!(
            "P4C_PLANAR_GPU_FOLLOWUP_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }

    #[cfg(target_os = "windows")]
    #[derive(Debug, Clone, Copy)]
    struct IntegratedFrameTiming {
        total_ns: u64,
        transport_and_ready_ns: u64,
        upload_enqueue_ns: u64,
        plan_encode_and_gpu_fence_ns: u64,
    }

    #[cfg(target_os = "windows")]
    #[derive(Debug, Default)]
    struct IntegratedSamples {
        total_ns: Vec<u64>,
        transport_and_ready_ns: Vec<u64>,
        upload_enqueue_ns: Vec<u64>,
        plan_encode_and_gpu_fence_ns: Vec<u64>,
    }

    #[cfg(target_os = "windows")]
    impl IntegratedSamples {
        fn with_capacity(capacity: usize) -> Self {
            Self {
                total_ns: Vec::with_capacity(capacity),
                transport_and_ready_ns: Vec::with_capacity(capacity),
                upload_enqueue_ns: Vec::with_capacity(capacity),
                plan_encode_and_gpu_fence_ns: Vec::with_capacity(capacity),
            }
        }

        fn push(&mut self, timing: IntegratedFrameTiming) {
            self.total_ns.push(timing.total_ns);
            self.transport_and_ready_ns
                .push(timing.transport_and_ready_ns);
            self.upload_enqueue_ns.push(timing.upload_enqueue_ns);
            self.plan_encode_and_gpu_fence_ns
                .push(timing.plan_encode_and_gpu_fence_ns);
        }

        fn extend(&mut self, other: &Self) {
            self.total_ns.extend_from_slice(&other.total_ns);
            self.transport_and_ready_ns
                .extend_from_slice(&other.transport_and_ready_ns);
            self.upload_enqueue_ns
                .extend_from_slice(&other.upload_enqueue_ns);
            self.plan_encode_and_gpu_fence_ns
                .extend_from_slice(&other.plan_encode_and_gpu_fence_ns);
        }
    }

    #[cfg(target_os = "windows")]
    fn elapsed_ns(start: std::time::Instant) -> u64 {
        u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    #[cfg(target_os = "windows")]
    fn integrated_stats(samples: &[u64]) -> serde_json::Value {
        assert!(!samples.is_empty(), "an integrated timing block was empty");
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        serde_json::json!({
            "count": sorted.len(),
            "min_ns": sorted[0],
            "p50_ns": percentile_ns(&sorted, 50),
            "p95_ns": percentile_ns(&sorted, 95),
            "p99_ns": percentile_ns(&sorted, 99),
            "max_ns": sorted[sorted.len() - 1],
        })
    }

    #[cfg(target_os = "windows")]
    fn integrated_samples_json(samples: &IntegratedSamples) -> serde_json::Value {
        serde_json::json!({
            "stats": {
                "total": integrated_stats(&samples.total_ns),
                "transport_and_ready": integrated_stats(&samples.transport_and_ready_ns),
                "upload_enqueue": integrated_stats(&samples.upload_enqueue_ns),
                "plan_encode_and_gpu_fence": integrated_stats(&samples.plan_encode_and_gpu_fence_ns),
            },
            "raw_ns": {
                "total": &samples.total_ns,
                "transport_and_ready": &samples.transport_and_ready_ns,
                "upload_enqueue": &samples.upload_enqueue_ns,
                "plan_encode_and_gpu_fence": &samples.plan_encode_and_gpu_fence_ns,
            },
        })
    }

    #[cfg(target_os = "windows")]
    fn integrated_p99(samples: &IntegratedSamples) -> u64 {
        let mut sorted = samples.total_ns.clone();
        sorted.sort_unstable();
        percentile_ns(&sorted, 99)
    }

    #[cfg(target_os = "windows")]
    fn integrated_plan(
        layer: &crate::layers::Layer,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> crate::evaluated_frame::EvaluatedFramePlan {
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };

        let modulation = crate::modulation::ModMatrix::new().frame(1);
        let master_effects = crate::effects::params::EffectUniforms::default();
        let master_transform = crate::spatial::SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(width, height, time_seconds),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(layer.layer_id(), 0, layer.width, layer.height),
                effects: &layer.effects,
                transform: &layer.transform,
                opacity: layer.opacity,
                mosh_send: layer.mosh_send,
                speed: layer.speed,
                fps: layer.fps,
                blend_mode: layer.blend_mode,
                visible: layer.visible,
                paused: layer.paused,
                bypass_master_fx: layer.bypass_master_fx,
                bypass_temporal_fx: layer.bypass_temporal_fx,
                pattern: None,
            }],
        )
    }

    /// Run one production-shaped accepted video frame from the authoritative
    /// transport tick through threaded-decoder harvest, the real layer upload
    /// seam, immutable frame evaluation, the complete default Exact compositor
    /// and opaque audience resolve, queue submission, and a GPU completion
    /// fence. The P4c planar conversion submits its own command buffer inside
    /// `upload_ready_media_frame`; queue ordering makes the final fence cover
    /// that work as well as the audience render.
    #[cfg(target_os = "windows")]
    fn run_integrated_frame(
        renderer: &mut crate::renderer::state::Renderer,
        layer: &mut crate::layers::Layer,
        policy: PlanarDeliveryPolicy,
        tick_ordinal: &mut u64,
    ) -> IntegratedFrameTiming {
        let total_started = std::time::Instant::now();
        let selection = {
            let mut attempts = 0_u8;
            loop {
                attempts = attempts.saturating_add(1);
                *tick_ordinal = tick_ordinal.saturating_add(1);
                let selection = layer
                    .apply_transport_tick_with_overrides(
                        crate::transport::ProgramTransportTick {
                            delta_seconds: 1.0 / 30.0,
                            program_beat: *tick_ordinal as f64 / 15.0,
                            program_running: true,
                            media_running: true,
                            ..Default::default()
                        },
                        1.0,
                        Some(30.0),
                    )
                    .expect("production transport selection");
                if selection.sample_due {
                    break selection;
                }
                assert!(attempts < 4, "30 fps fixture failed to schedule a sample");
            }
        };

        let expected_format = match policy {
            PlanarDeliveryPolicy::LegacyRgba => crate::video::DecodedPixelFormat::PackedRgba8,
            PlanarDeliveryPolicy::MetadataManaged => {
                crate::video::DecodedPixelFormat::PlanarYuv420p8
            }
        };
        let ready_deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut frame = loop {
            if let Some(frame) = layer.take_ready_media_frame().expect("production harvest") {
                // A same-generation continuous decode may finish while the
                // next absolute selection is queued; that is exactly the
                // newest-ready behavior Main consumes. An older generation is
                // never admissible across a loop/seek discontinuity. A layer
                // opened under the legacy default may also have one packed
                // seed in flight when the managed policy is first authored;
                // warm-up drains that transition before any sample is accepted.
                if frame.source_generation == selection.generation
                    && frame.rgba.layout().format == expected_format
                {
                    break frame;
                }
            }
            assert!(
                std::time::Instant::now() < ready_deadline,
                "no matching decoded frame arrived within five seconds"
            );
            std::thread::sleep(Duration::from_micros(250));
        };
        let transport_and_ready_ns = elapsed_ns(total_started);

        let upload_started = std::time::Instant::now();
        let renderer_generation = renderer.renderer_generation();
        layer
            .upload_ready_media_frame(
                &renderer.device,
                &renderer.queue,
                renderer_generation,
                &mut frame,
            )
            .expect("production layer upload");
        let upload_enqueue_ns = elapsed_ns(upload_started);
        assert_eq!(
            layer.delivery_active_planar(),
            policy == PlanarDeliveryPolicy::MetadataManaged,
            "published delivery state diverged from the accepted payload"
        );

        let render_started = std::time::Instant::now();
        let plan = integrated_plan(
            layer,
            renderer.output_width,
            renderer.output_height,
            frame.source_seconds as f32,
        );
        let resources =
            crate::renderer::state::LiveFrameResources::capture(std::slice::from_ref(&*layer));
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("P4c integrated total-frame encoder"),
            });
        renderer
            .render_evaluated_frame(&mut encoder, &resources, &plan)
            .expect("production Exact composition");
        renderer.render_temporal_with_dt(&mut encoder, plan.temporal(), 1.0 / 30.0, true);
        renderer.render_opaque_output(&mut encoder);
        renderer.queue.submit(std::iter::once(encoder.finish()));
        renderer.commit_temporal_frame();
        renderer
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("integrated GPU completion fence");

        for _ in 0..4 {
            if let Some(fault) = layer.poll_upload_validation(renderer_generation) {
                panic!(
                    "integrated upload validation fault: {}",
                    fault.operator_message()
                );
            }
            if layer.upload_validation_snapshot().pending == 0 {
                break;
            }
            let _ = renderer.device.poll(wgpu::PollType::Poll);
        }
        let validation = layer.upload_validation_snapshot();
        assert_eq!(validation.pending, 0, "upload validation did not drain");
        assert_eq!(validation.faults, 0, "upload validation faulted");

        IntegratedFrameTiming {
            total_ns: elapsed_ns(total_started),
            transport_and_ready_ns,
            upload_enqueue_ns,
            plan_encode_and_gpu_fence_ns: elapsed_ns(render_started),
        }
    }

    #[cfg(target_os = "windows")]
    fn measure_integrated_block(
        renderer: &mut crate::renderer::state::Renderer,
        layer: &mut crate::layers::Layer,
        policy: PlanarDeliveryPolicy,
        tick_ordinal: &mut u64,
        minimum_duration: Duration,
        minimum_samples: usize,
    ) -> IntegratedSamples {
        let started = std::time::Instant::now();
        let mut samples = IntegratedSamples::with_capacity(minimum_samples);
        while started.elapsed() < minimum_duration || samples.total_ns.len() < minimum_samples {
            samples.push(run_integrated_frame(renderer, layer, policy, tick_ordinal));
        }
        samples
    }

    /// Prime the renderer's deliberately single-live-set texture bind-group
    /// cache after each AB/BA cohort switch, then prove the measured block is
    /// allocation-free. The transition frame is outside the sample vectors.
    #[cfg(target_os = "windows")]
    fn measure_warmed_integrated_block(
        renderer: &mut crate::renderer::state::Renderer,
        layer: &mut crate::layers::Layer,
        policy: PlanarDeliveryPolicy,
        tick_ordinal: &mut u64,
        minimum_duration: Duration,
        minimum_samples: usize,
    ) -> IntegratedSamples {
        let _ = run_integrated_frame(renderer, layer, policy, tick_ordinal);
        let warmed = renderer.core_gpu_object_construction_snapshot();
        let samples = measure_integrated_block(
            renderer,
            layer,
            policy,
            tick_ordinal,
            minimum_duration,
            minimum_samples,
        );
        let measured_delta = renderer
            .core_gpu_object_construction_snapshot()
            .delta_since(warmed);
        assert_eq!(
            measured_delta.total(),
            0,
            "a post-transition measured block constructed GPU objects: {measured_delta:?}"
        );
        samples
    }

    #[cfg(target_os = "windows")]
    fn sha256_file(path: &std::path::Path) -> String {
        use sha2::{Digest, Sha256};

        let digest = Sha256::digest(std::fs::read(path).expect("read generated source for digest"));
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[cfg(target_os = "windows")]
    fn command_version(program: &std::ffi::OsStr, args: &[&str]) -> String {
        std::process::Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }

    /// The P4c follow-on default/auto-selection gate. Unlike the Phase-A seam
    /// receipt, this fixture times one accepted production-shaped frame from
    /// transport/decode readiness through final opaque audience completion.
    /// It deliberately excludes surface presentation and photon time, and it
    /// neither asserts nor hides the outcome: the receipt records a truthful
    /// pass/fail independently for 720p and 1080p and leaves `legacy_rgba` as
    /// the default when either source regresses.
    ///
    /// The evidence run is fixed at 300 warm accepted frames per cohort, five
    /// paired AB/BA runs per raster, and ten aggregate measurement minutes.
    /// `COLLIDE_O_SCOPE_P4C_SMOKE=1` only checks the machinery quickly and is
    /// branded ineligible for the default gate in its receipt.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "release-only Windows GPU + FFmpeg campaign; writes an untracked integrated p99 receipt"]
    fn gpu_planar_integrated_total_frame_p99_writes_the_receipt() {
        use std::sync::Arc;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        if cfg!(debug_assertions) {
            panic!("integrated timing evidence must come from --release");
        }
        ffmpeg_next::init().ok();

        let smoke = match std::env::var("COLLIDE_O_SCOPE_P4C_SMOKE") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            Ok(value) => panic!("COLLIDE_O_SCOPE_P4C_SMOKE accepts only '1'; got {value:?}"),
            Err(error) => panic!("could not read COLLIDE_O_SCOPE_P4C_SMOKE: {error}"),
        };
        let warm_frames = if smoke { 3 } else { 300 };
        let paired_runs = 5_usize;
        let total_measurement = if smoke {
            Duration::from_millis(250)
        } else {
            Duration::from_secs(10 * 60)
        };
        let minimum_samples_per_block = if smoke { 2 } else { 120 };

        let media = CandidateMedia::generate().expect("generate integrated sources");
        let source_count = media.sources.len();
        assert_eq!(source_count, 2, "the gate requires exactly two sources");
        let blocks = source_count * paired_runs * 2;
        let block_duration =
            Duration::from_secs_f64(total_measurement.as_secs_f64() / blocks as f64);

        let mut event_loop_builder = winit::event_loop::EventLoop::<()>::builder();
        event_loop_builder.with_any_thread(true);
        let event_loop = event_loop_builder.build().expect("fixture event loop");

        let mut source_reports = Vec::with_capacity(source_count);
        let mut all_sources_pass = true;
        for (label, path, width, height) in &media.sources {
            #[allow(deprecated)]
            let window = Arc::new(
                event_loop
                    .create_window(
                        winit::window::Window::default_attributes()
                            .with_title(format!("P4c {label} total-frame fixture"))
                            .with_visible(false)
                            .with_inner_size(winit::dpi::PhysicalSize::new(64, 64)),
                    )
                    .expect("hidden fixture window"),
            );
            let mut renderer = crate::renderer::state::Renderer::new(window, *width, *height)
                .expect("production renderer");
            let adapter = renderer.device.adapter_info();
            let clip_text = path.to_string_lossy().into_owned();
            let mut legacy = crate::layers::Layer::new_with_media_policy(
                &clip_text,
                &renderer.device,
                &crate::media_safety::MediaSafetyPolicy::safe(),
            )
            .expect("legacy fixture layer");
            let mut managed = crate::layers::Layer::new_with_media_policy(
                &clip_text,
                &renderer.device,
                &crate::media_safety::MediaSafetyPolicy::safe(),
            )
            .expect("managed fixture layer");
            legacy.set_delivery_policy(PlanarDeliveryPolicy::LegacyRgba);
            managed.set_delivery_policy(PlanarDeliveryPolicy::MetadataManaged);

            let mut legacy_tick = 0_u64;
            let mut managed_tick = 0_u64;
            for _ in 0..warm_frames {
                let _ = run_integrated_frame(
                    &mut renderer,
                    &mut legacy,
                    PlanarDeliveryPolicy::LegacyRgba,
                    &mut legacy_tick,
                );
            }
            for _ in 0..warm_frames {
                let _ = run_integrated_frame(
                    &mut renderer,
                    &mut managed,
                    PlanarDeliveryPolicy::MetadataManaged,
                    &mut managed_tick,
                );
            }
            let warmed_objects = renderer.core_gpu_object_construction_snapshot();

            let mut legacy_all = IntegratedSamples::default();
            let mut managed_all = IntegratedSamples::default();
            let mut run_reports = Vec::with_capacity(paired_runs);
            let mut every_paired_run_passed = true;
            for run_index in 0..paired_runs {
                let legacy_first = run_index % 2 == 0;
                let (legacy_run, managed_run) = if legacy_first {
                    let legacy_run = measure_warmed_integrated_block(
                        &mut renderer,
                        &mut legacy,
                        PlanarDeliveryPolicy::LegacyRgba,
                        &mut legacy_tick,
                        block_duration,
                        minimum_samples_per_block,
                    );
                    let managed_run = measure_warmed_integrated_block(
                        &mut renderer,
                        &mut managed,
                        PlanarDeliveryPolicy::MetadataManaged,
                        &mut managed_tick,
                        block_duration,
                        minimum_samples_per_block,
                    );
                    (legacy_run, managed_run)
                } else {
                    let managed_run = measure_warmed_integrated_block(
                        &mut renderer,
                        &mut managed,
                        PlanarDeliveryPolicy::MetadataManaged,
                        &mut managed_tick,
                        block_duration,
                        minimum_samples_per_block,
                    );
                    let legacy_run = measure_warmed_integrated_block(
                        &mut renderer,
                        &mut legacy,
                        PlanarDeliveryPolicy::LegacyRgba,
                        &mut legacy_tick,
                        block_duration,
                        minimum_samples_per_block,
                    );
                    (legacy_run, managed_run)
                };
                let legacy_p99 = integrated_p99(&legacy_run);
                let managed_p99 = integrated_p99(&managed_run);
                let run_passed = managed_p99 <= legacy_p99;
                every_paired_run_passed &= run_passed;
                run_reports.push(serde_json::json!({
                    "run": run_index + 1,
                    "order": if legacy_first { "legacy_then_managed" } else { "managed_then_legacy" },
                    "legacy_rgba": integrated_samples_json(&legacy_run),
                    "metadata_managed": integrated_samples_json(&managed_run),
                    "managed_total_p99_no_worse": run_passed,
                    "managed_minus_legacy_p99_ns": i128::from(managed_p99) - i128::from(legacy_p99),
                }));
                legacy_all.extend(&legacy_run);
                managed_all.extend(&managed_run);
            }

            let final_objects = renderer.core_gpu_object_construction_snapshot();
            let warmed_object_delta = final_objects.delta_since(warmed_objects);
            let legacy_p99 = integrated_p99(&legacy_all);
            let managed_p99 = integrated_p99(&managed_all);
            let aggregate_passed = managed_p99 <= legacy_p99;
            let source_passed = aggregate_passed && every_paired_run_passed;
            all_sources_pass &= source_passed;
            let packed_bytes = u64::from(*width) * u64::from(*height) * 4;
            let planar_bytes = u64::from(*width) * u64::from(*height) * 3 / 2;
            source_reports.push(serde_json::json!({
                "label": label,
                "width": width,
                "height": height,
                "source_sha256": sha256_file(path),
                "adapter": {
                    "name": adapter.name,
                    "backend": format!("{:?}", adapter.backend),
                    "device_type": format!("{:?}", adapter.device_type),
                    "vendor": adapter.vendor,
                    "device": adapter.device,
                    "driver": adapter.driver,
                    "driver_info": adapter.driver_info,
                },
                "warm_accepted_frames_per_policy": warm_frames,
                "paired_runs": run_reports,
                "aggregate": {
                    "legacy_rgba": integrated_samples_json(&legacy_all),
                    "metadata_managed": integrated_samples_json(&managed_all),
                    "managed_total_p99_no_worse": aggregate_passed,
                    "managed_minus_legacy_p99_ns": i128::from(managed_p99) - i128::from(legacy_p99),
                },
                "fallback_frames": {
                    "legacy_wrong_format": 0,
                    "managed_not_planar": 0,
                },
                "bytes_per_accepted_source_frame": {
                    "legacy_rgba8": packed_bytes,
                    "metadata_managed_yuv420p8": planar_bytes,
                    "staging_reduction_percent": 100.0 - planar_bytes as f64 * 100.0 / packed_bytes as f64,
                },
                "out_of_sample_cohort_transition_object_delta": {
                    "buffers": warmed_object_delta.buffers,
                    "bind_groups": warmed_object_delta.bind_groups,
                    "pipelines": warmed_object_delta.pipelines,
                    "textures": warmed_object_delta.textures,
                    "samplers": warmed_object_delta.samplers,
                    "law": "the renderer intentionally retains only the current LiveFrameResources stable-id set; one transition frame primes each AB/BA cohort, and every measured block separately asserts zero GPU-object construction",
                },
                "gate": {
                    "aggregate_p99_non_regression": aggregate_passed,
                    "all_five_paired_runs_non_regressing": every_paired_run_passed,
                    "source_passed": source_passed,
                },
            }));
        }

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                .unwrap_or_else(|| "unknown".to_owned())
        };
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let ffmpeg = crate::host_paths::ffmpeg();
        let receipt = serde_json::json!({
            "schema": "collide-o-scope-p4c-integrated-total-frame-p99-receipt/1",
            "run_kind": if smoke { "smoke_ineligible_for_default_gate" } else { "evidence" },
            "command": "cargo test --locked --release --bin collide-o-scope gpu_planar_integrated_total_frame_p99_writes_the_receipt -- --ignored --nocapture --test-threads=1",
            "smoke_command": "$env:COLLIDE_O_SCOPE_P4C_SMOKE='1'; cargo test --locked --release --bin collide-o-scope gpu_planar_integrated_total_frame_p99_writes_the_receipt -- --ignored --nocapture --test-threads=1; Remove-Item Env:COLLIDE_O_SCOPE_P4C_SMOKE",
            "measured_at": {
                "commit": git(&["rev-parse", "HEAD"]),
                "tree": git(&["rev-parse", "HEAD^{tree}"]),
                "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
                "working_tree": match git(&["status", "--porcelain"]).as_str() {
                    "unknown" => "unknown",
                    "" => "clean",
                    _ => "dirty",
                },
            },
            "host": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "build_profile": "release",
                "rustc": command_version(&rustc, &["--version"]),
                "ffmpeg": command_version(ffmpeg.as_os_str(), &["-version"]),
            },
            "method": {
                "boundary": "before ProgramTransportTick and threaded-decoder readiness; through Layer::upload_ready_media_frame, EvaluatedFramePlan::evaluate, LiveFrameResources::capture, Renderer::render_evaluated_frame, default Temporal, opaque audience resolve, queue submit, Device::poll wait fence, and upload-scope drain",
                "excluded": [
                    "cold renderer/layer/decoder/pipeline/resource construction",
                    "window surface acquisition and presentation",
                    "display scanout and photon time",
                    "unrelated Main control, UI, recorder, and optional-effect work",
                ],
                "source_schedule": "two independently generated H.264 yuv420p 30 fps clips with declared tv/bt709/bt709/left metadata; production absolute transport selections advance at 30 accepted ticks per second",
                "comparison": "one architectural variable: identical source/raster/default Exact program, separate warmed layers and decoders, legacy_rgba versus metadata_managed, AB/BA order alternating across five paired runs",
                "warm_accepted_frames_per_policy_per_source": warm_frames,
                "paired_runs_per_source": paired_runs,
                "aggregate_measurement_seconds": total_measurement.as_secs_f64(),
                "minimum_samples_per_block": minimum_samples_per_block,
                "raw_samples_included": true,
            },
            "sources_measured": source_reports,
            "default_gate": {
                "eligible_run_shape": !smoke,
                "requires_each_source_aggregate_and_all_five_pairs": true,
                "all_sources_passed": all_sources_pass,
                "passed": !smoke && all_sources_pass,
                "decision_law": "legacy_rgba remains the Rust/serde/default-selection law unless this evidence run passes independently at 1280x720 and 1920x1080; a smoke run or any regression is a truthful negative and cannot authorize a default flip",
            },
        });
        std::fs::create_dir_all("target").expect("create ignored receipt directory");
        let receipt_path = "target/p4c-planar-integrated-total-frame-p99-receipt.json";
        std::fs::write(
            receipt_path,
            format!("{}\n", serde_json::to_string_pretty(&receipt).unwrap()),
        )
        .expect("write integrated P4c p99 receipt");
        println!(
            "P4C_PLANAR_INTEGRATED_TOTAL_FRAME_P99_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }
}
