//! The evaluation-only GPU twin of the P4c planar CPU oracle.
//!
//! This module reopens P4c exactly the way its stop receipt prescribed: "a
//! dedicated renderer branch and the audit's 720p/1080p two-source CPU/GPU
//! equality plus p95/p99 fixture" — and deliberately nothing further. It is
//! the `hw_decode`/full-16 shape: **measurement-only**. No production decode,
//! upload, or render path constructs a converter; the only constructors are
//! the opt-in equality battery, the opt-in candidate-measurement fixture
//! (which regenerates the tracked
//! `docs/evidence/p4c-planar-gpu-candidate-receipt.json` under the S2-receipt
//! law), and the hosted contract tests. Promotion — a decoder that selects
//! planar delivery, pooled staging, patch policy, ledger accounting — remains
//! a separate operator-decided tranche taken after reading the receipt.
//!
//! The conversion law is not restated here. The GPU uniforms derive from the
//! exact [`CpuConversionContract`] the CPU oracle consumes, so admission,
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
//! apart and nothing else may.
//!
//! The shader is embedded as a module constant rather than a file under
//! `src/shaders/`, deliberately: `build.rs` hashes that directory into the
//! production shader-bundle identity, and an evaluation-only shader is not
//! part of the production bundle. Validation happens where the shader runs —
//! at pipeline creation inside the opt-in fixtures.

// The converter's only constructors are the opt-in fixtures and the hosted
// contract tests — the S10a discipline: name the consumer honestly rather
// than fake a premature integration.
#![allow(
    dead_code,
    reason = "P4c stays measured-before-promotion; production consumers are forbidden until the candidate receipt clears the audit's promotion gate"
)]

use bytemuck::{Pod, Zeroable};

use super::planar::{
    chroma_sample_offset, CpuConversionContract, PlanarConversionError, PlanarImageLayout,
    PlanarImagePayload, PlanarPixelFormat, PlanarPlaneKind,
};
use super::{SourceColorDescriptor, SourceColorRange};

/// The complete conversion pass, embedded so the production shader bundle
/// keeps its identity. The fragment mirrors the CPU oracle expression for
/// expression; see the module documentation for the tolerance argument.
const PLANAR_CONVERT_WGSL: &str = r#"
struct PlanarUniforms {
    size: vec2<u32>,
    format: u32,
    bit_depth: u32,
    range_full: u32,
    _pad0: u32,
    chroma_offset: vec2<f32>,
    kr: f32,
    kb: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: PlanarUniforms;
@group(0) @binding(1) var luma_plane: texture_2d<u32>;
@group(0) @binding(2) var chroma_a: texture_2d<u32>;
@group(0) @binding(3) var chroma_b: texture_2d<u32>;

@vertex
fn vs_convert(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn luma_code(pixel: vec2<u32>) -> f32 {
    let word = textureLoad(luma_plane, vec2<i32>(pixel), 0).r;
    if uniforms.format == 2u {
        return f32(word >> 6u);
    }
    return f32(word);
}

fn chroma_pair_at(texel: vec2<i32>) -> vec2<f32> {
    if uniforms.format == 0u {
        return vec2<f32>(
            f32(textureLoad(chroma_a, texel, 0).r),
            f32(textureLoad(chroma_b, texel, 0).r),
        );
    }
    let pair = textureLoad(chroma_a, texel, 0).rg;
    if uniforms.format == 2u {
        return vec2<f32>(f32(pair.x >> 6u), f32(pair.y >> 6u));
    }
    return vec2<f32>(f32(pair.x), f32(pair.y));
}

fn chroma_code(pixel: vec2<u32>) -> vec2<f32> {
    let plane_size = textureDimensions(chroma_a);
    let plane_w = f32(plane_size.x);
    let plane_h = f32(plane_size.y);
    let x = clamp(
        (f32(pixel.x) - uniforms.chroma_offset.x) / 2.0,
        0.0,
        plane_w - 1.0,
    );
    let y = clamp(
        (f32(pixel.y) - uniforms.chroma_offset.y) / 2.0,
        0.0,
        plane_h - 1.0,
    );
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = min(x0 + 1.0, plane_w - 1.0);
    let y1 = min(y0 + 1.0, plane_h - 1.0);
    let tx = x - x0;
    let ty = y - y0;
    let c00 = chroma_pair_at(vec2<i32>(i32(x0), i32(y0)));
    let c10 = chroma_pair_at(vec2<i32>(i32(x1), i32(y0)));
    let c01 = chroma_pair_at(vec2<i32>(i32(x0), i32(y1)));
    let c11 = chroma_pair_at(vec2<i32>(i32(x1), i32(y1)));
    let top = c00 * (1.0 - tx) + c10 * tx;
    let bottom = c01 * (1.0 - tx) + c11 * tx;
    let max_code = f32((1u << uniforms.bit_depth) - 1u);
    return clamp(
        top * (1.0 - ty) + bottom * ty,
        vec2<f32>(0.0, 0.0),
        vec2<f32>(max_code, max_code),
    );
}

@fragment
fn fs_convert(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(position.x), u32(position.y));
    let y_code = luma_code(pixel);
    let chroma = chroma_code(pixel);
    let bd = uniforms.bit_depth;
    let scale = f32(1u << (bd - 8u));
    let max_code = f32((1u << bd) - 1u);
    var y: f32;
    var cb: f32;
    var cr: f32;
    if uniforms.range_full == 1u {
        y = y_code / max_code;
        cb = (chroma.x - f32(1u << (bd - 1u))) / max_code;
        cr = (chroma.y - f32(1u << (bd - 1u))) / max_code;
    } else {
        y = (y_code - 16.0 * scale) / (219.0 * scale);
        cb = (chroma.x - 128.0 * scale) / (224.0 * scale);
        cr = (chroma.y - 128.0 * scale) / (224.0 * scale);
    }
    let kr = uniforms.kr;
    let kb = uniforms.kb;
    let kg = 1.0 - kr - kb;
    let red = y + (2.0 - 2.0 * kr) * cr;
    let blue = y + (2.0 - 2.0 * kb) * cb;
    let green = y - kb * (2.0 - 2.0 * kb) / kg * cb - kr * (2.0 - 2.0 * kr) / kg * cr;
    return vec4<f32>(
        clamp(vec3<f32>(red, green, blue), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)),
        1.0,
    );
}
"#;

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

impl PlanarPlaneTextures {
    pub const fn layout(&self) -> PlanarImageLayout {
        self.layout
    }
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

    /// The S10a discipline as a source audit: the prototype has no
    /// production consumer, and a reference appearing anywhere else in the
    /// tree is a promotion that must instead go through the audit's gate.
    #[test]
    fn no_production_module_consumes_the_planar_gpu_prototype() {
        let mut pending = vec![std::path::PathBuf::from("src")];
        let mut offenders = Vec::new();
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(&directory).expect("walk src") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                if path.extension().is_none_or(|extension| extension != "rs") {
                    continue;
                }
                let relative = path.to_string_lossy().replace('\\', "/");
                // The prototype itself, its module declaration, and the
                // contract module whose documentation names its GPU twin.
                if relative == "src/video/planar_gpu.rs"
                    || relative == "src/video/mod.rs"
                    || relative == "src/video/planar.rs"
                {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("read source");
                if source.contains("planar_gpu") {
                    offenders.push(relative);
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "planar_gpu is measurement-only; production references found in {offenders:?}"
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
                let output = std::process::Command::new(crate::host_paths::ffmpeg())
                    .args([
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        &format!("testsrc2=size={width}x{height}:rate=30:duration=8.5"),
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
                        // The generic -color_* options reach the container,
                        // but libx264's VUI carries primaries/transfer only
                        // through its own parameter surface; without these
                        // the decoder honestly reports them unspecified and
                        // admission fails.
                        "-x264-params",
                        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
                        "-y",
                    ])
                    .arg(&path)
                    .output()
                    .map_err(|error| error.to_string())?;
                if !output.status.success() {
                    let error = String::from_utf8_lossy(&output.stderr).into_owned();
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
        label: String,
        width: u32,
        height: u32,
        measured_frames: usize,
        packed_delivery_ns: Vec<u64>,
        planar_delivery_ns: Vec<u64>,
        spot_payloads: Vec<(usize, PlanarImagePayload)>,
        first_packed_rgba: Vec<u8>,
        descriptor: SourceColorDescriptor,
    }

    /// Decode one candidate source with the production-shaped software loop
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
            super::super::planar::prototype_delivery_decision(
                PlanarDeliveryPolicy::MetadataManaged,
                PlanarPixelFormat::Yuv420p8,
                descriptor,
                field_order,
            ),
            PlanarDeliveryDecision::PrototypePlanar(PlanarPixelFormat::Yuv420p8),
            "{label}: the candidate source must pass the real admission law \
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
            label: label.to_owned(),
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
            "{label}: candidate source decoded too few frames"
        );
        measurement
    }

    /// The reopened P4c candidate measurement: the audit's 720p/1080p
    /// two-source CPU/GPU equality plus p95/p99 fixture, regenerating the
    /// tracked `docs/evidence/p4c-planar-gpu-candidate-receipt.json` in
    /// place (the S2-receipt law: a changed receipt after an opt-in run is a
    /// new measurement on new hardware; commit it). Performance numbers are
    /// recorded, never asserted — a truthful negative result is a valid
    /// completion, exactly as it was for D3D11VA.
    #[test]
    #[ignore = "requires a GPU adapter and an ffmpeg executable; emits the P4c candidate receipt"]
    fn gpu_planar_delivery_candidate_measures_conversion_upload_and_writes_the_receipt() {
        use std::time::Instant;

        if cfg!(debug_assertions) {
            panic!("timing evidence must come from --release; debug timings are not evidence");
        }
        ffmpeg_next::init().ok();

        let Some((device, queue, adapter)) = acquire_device() else {
            panic!("the candidate receipt requires a GPU adapter");
        };
        let converter = PlanarGpuConverter::new(&device);
        let media = CandidateMedia::generate().expect("generate candidate sources");

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
                "admission": "PrototypePlanar(Yuv420p8) through prototype_delivery_decision on the production-frozen descriptor",
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
            "schema": "collide-o-scope-p4c-planar-gpu-candidate-receipt/1",
            "command": "cargo test --locked --release gpu_planar_delivery_candidate_measures_conversion_upload_and_writes_the_receipt -- --ignored --nocapture",
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
            "scope": "evaluation-only: the GPU converter is constructed by the opt-in fixtures alone; no decoder selects planar delivery, no patch surface changed, and promotion remains gated on the audit's integrated two-source matrix without worsening total-frame p99",
            "sources": "two generated H.264 yuv420p clips (testsrc2, 30 fps) with declared tv/bt709/bt709/left metadata; the real decoder's frozen descriptor drives admission and conversion",
            "conversion_law": "source_encoded_sdr_rgba8_no_gamut_mapping — the CPU oracle's law; the GPU twin derives uniforms from the same CpuConversionContract",
            "sources_measured": source_reports,
        });
        std::fs::write(
            "docs/evidence/p4c-planar-gpu-candidate-receipt.json",
            format!("{}\n", serde_json::to_string_pretty(&receipt).unwrap()),
        )
        .expect("write the P4c candidate receipt");
        println!(
            "P4C_PLANAR_GPU_CANDIDATE_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }
}
