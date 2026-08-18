//! S2 probe: is one pass that simultaneously samples eight textures portable
//! under the device floor collide-o-scope already requires?
//!
//! This is a throwaway investigation fixture. It ships no feature, touches no
//! product module, and changes no rack admission constant. It answers exactly
//! one scheduling question and emits a named receipt.
//!
//! # Method
//!
//! The production renderer creates its device at `src/renderer/state.rs:2892`
//! with `required_features: wgpu::Features::empty()` and
//! `required_limits: wgpu::Limits::default()`. That request is the device floor.
//! Two properties follow from it, and both matter here:
//!
//! 1. wgpu **rejects** `request_device` on any adapter that cannot satisfy the
//!    requested limits. An adapter below the floor therefore cannot run
//!    collide-o-scope at all.
//! 2. wgpu **caps** validation at the *requested* limits, not at the adapter's
//!    raw capability. A device created with `Limits::default()` behaves like a
//!    device that has exactly 16 sampled textures per shader stage, even on an
//!    adapter that reports hundreds.
//!
//! Property 2 is what makes this probe a portability proof rather than a
//! hardware anecdote. Every fixture below requests exactly `Limits::default()`,
//! so a success is a success *inside the floor's budget* and generalizes to any
//! conforming adapter. The negative control at 17 textures demonstrates that the
//! cap is genuinely enforced on this device, which is what earns the positive
//! result its meaning.

use std::io::Write as _;

use sha2::{Digest, Sha256};

/// Matches `RACK_TEXTURE_FORMAT` in `src/renderer/rack.rs:29`.
const PROBE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The claim under test.
const PROBE_TEXTURE_COUNT: u32 = 8;

/// `wgpu::Limits::default().max_sampled_textures_per_shader_stage` in
/// wgpu-types 29.0.3 (`src/limits.rs:383`).
const FLOOR_SAMPLED_TEXTURES: u32 = 16;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct FloorHarness {
    adapter_info: wgpu::AdapterInfo,
    adapter_reported: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl FloorHarness {
    /// Replicates the production device request byte for byte.
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for the eight-texture floor probe");

        let adapter_info = adapter.get_info();
        let adapter_reported = adapter.limits().max_sampled_textures_per_shader_stage;

        // Exactly the production request: state.rs:2892.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Eight-texture floor probe device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device at the collide-o-scope floor");

        Self {
            adapter_info,
            adapter_reported,
            device,
            queue,
        }
    }

    /// The limit this device will actually validate against.
    fn enforced_sampled_textures(&self) -> u32 {
        self.device.limits().max_sampled_textures_per_shader_stage
    }

    /// A 1x1 Rgba16Float texture whose red channel carries `red`.
    fn texture(&self, red: f32, label: &str) -> wgpu::TextureView {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PROBE_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut bytes = Vec::with_capacity(8);
        for channel in [red, 0.0, 0.0, 1.0] {
            bytes.extend_from_slice(&f32_to_f16(channel).to_le_bytes());
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Builds a fragment-visible layout with `count` sampled textures, one
    /// shared filtering sampler, and one uniform buffer. Returns the validation
    /// error instead of panicking, so the caller can assert on refusal.
    fn try_layout(&self, count: u32, label: &str) -> Result<wgpu::BindGroupLayout, wgpu::Error> {
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..count)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: count,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: count + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        match pollster::block_on(scope.pop()) {
            Some(error) => Err(error),
            None => Ok(layout),
        }
    }
}

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

/// One fragment shader that samples `count` distinct textures in a single pass.
///
/// Each tap is weighted by a value read from a uniform buffer, so no weight is
/// a compile-time constant and the compiler cannot fold, dedupe, or eliminate
/// any of the sampling operations. All `count` bindings are load-bearing.
fn probe_shader(count: u32) -> String {
    let mut source = String::new();
    for binding in 0..count {
        source.push_str(&format!(
            "@group(0) @binding({binding}) var tap_{binding}: texture_2d<f32>;\n"
        ));
    }
    source.push_str(&format!(
        "@group(0) @binding({}) var probe_sampler: sampler;\n",
        count
    ));
    source.push_str(&format!(
        "@group(0) @binding({}) var<uniform> weights: Weights;\n",
        count + 1
    ));
    source.push_str(&format!(
        "\nstruct Weights {{ lanes: array<vec4<f32>, {}> }};\n\n",
        count.div_ceil(4).max(1)
    ));

    source.push_str(
        "@vertex\n\
         fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {\n\
         \x20   let x = f32(i32(index) / 2) * 4.0 - 1.0;\n\
         \x20   let y = f32(i32(index) & 1) * 4.0 - 1.0;\n\
         \x20   return vec4<f32>(x, y, 0.0, 1.0);\n\
         }\n\n\
         @fragment\n\
         fn fs_main() -> @location(0) vec4<f32> {\n\
         \x20   let uv = vec2<f32>(0.5, 0.5);\n\
         \x20   var total = 0.0;\n",
    );
    for binding in 0..count {
        let lane = binding / 4;
        let channel = ["x", "y", "z", "w"][(binding % 4) as usize];
        source.push_str(&format!(
            "    total = total + textureSample(tap_{binding}, probe_sampler, uv).r \
             * weights.lanes[{lane}].{channel};\n"
        ));
    }
    source.push_str("    return vec4<f32>(total, 0.0, 0.0, 1.0);\n}\n");
    source
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The named receipt fixture. Renders one pass that simultaneously samples
/// eight textures on a device capped at the collide-o-scope floor, verifies
/// every tap reached the output, establishes the 16/17 boundary on the same
/// device, and writes the receipt.
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_eight_sampled_textures_in_one_pass_are_portable_under_the_device_floor() {
    let gpu = FloorHarness::new();

    // -- Precondition: the device really is capped at the documented floor. --
    assert_eq!(
        gpu.enforced_sampled_textures(),
        FLOOR_SAMPLED_TEXTURES,
        "requesting Limits::default() must cap validation at the floor, not at \
         the adapter's raw capability ({} reported)",
        gpu.adapter_reported
    );

    // -- Positive: eight textures, one bind group, one pass. --
    let layout = gpu
        .try_layout(PROBE_TEXTURE_COUNT, "eight-texture probe layout")
        .expect("eight fragment-visible sampled textures must fit the floor");

    // Tap i carries red = i + 1 and is weighted by 2^i, so the sum is a unique
    // fingerprint: dropping, duplicating, or reordering any tap changes it.
    let views: Vec<wgpu::TextureView> = (0..PROBE_TEXTURE_COUNT)
        .map(|index| gpu.texture((index + 1) as f32, "probe tap"))
        .collect();
    let weights: Vec<f32> = (0..PROBE_TEXTURE_COUNT)
        .map(|i| (1u32 << i) as f32)
        .collect();
    let expected: f32 = (0..PROBE_TEXTURE_COUNT)
        .map(|i| (i + 1) as f32 * (1u32 << i) as f32)
        .sum();
    assert_eq!(
        expected, 1793.0,
        "probe fingerprint is fixed by construction"
    );

    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("probe sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe weights"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut weight_bytes = Vec::with_capacity(32);
    for weight in &weights {
        weight_bytes.extend_from_slice(&weight.to_le_bytes());
    }
    gpu.queue.write_buffer(&uniform, 0, &weight_bytes);

    let mut bindings: Vec<wgpu::BindGroupEntry> = views
        .iter()
        .enumerate()
        .map(|(index, view)| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();
    bindings.push(wgpu::BindGroupEntry {
        binding: PROBE_TEXTURE_COUNT,
        resource: wgpu::BindingResource::Sampler(&sampler),
    });
    bindings.push(wgpu::BindGroupEntry {
        binding: PROBE_TEXTURE_COUNT + 1,
        resource: uniform.as_entire_binding(),
    });
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eight-texture probe bind group"),
        layout: &layout,
        entries: &bindings,
    });

    let shader_source = probe_shader(PROBE_TEXTURE_COUNT);
    let shader_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(shader_source.as_bytes());
        format!("{:x}", hasher.finalize())
    };

    // Naga validates the module here; a miscounted binding fails now.
    let shader_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eight-texture probe shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.as_str().into()),
        });
    assert!(
        pollster::block_on(shader_scope.pop()).is_none(),
        "the eight-tap shader must pass Naga validation at the floor"
    );

    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eight-texture probe pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

    let pipeline_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("eight-texture probe pipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: PROBE_TEXTURE_FORMAT,
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
    assert!(
        pollster::block_on(pipeline_scope.pop()).is_none(),
        "the eight-tap pipeline must build at the floor"
    );

    // -- Execute one pass and read the fingerprint back. --
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("eight-texture probe target"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: PROBE_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("eight-texture probe readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eight-texture probe encoder"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("eight-texture probe pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (send, receive) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = send.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU wait");
    receive.recv().expect("map callback").expect("map result");
    let observed = {
        let mapped = slice.get_mapped_range();
        f16_to_f32(u16::from_le_bytes([mapped[0], mapped[1]]))
    };
    staging.unmap();

    assert_eq!(
        observed, expected,
        "all eight taps must reach the output in one pass; 1793 is exact in \
         binary16 and any dropped or duplicated tap changes it"
    );

    // -- Boundary: the floor itself is reachable. --
    let sixteen = gpu.try_layout(FLOOR_SAMPLED_TEXTURES, "sixteen-texture layout");
    assert!(
        sixteen.is_ok(),
        "the floor's own maximum must be reachable: {:?}",
        sixteen.err()
    );

    // -- Negative control: the cap is genuinely enforced on this device. --
    let over = gpu.try_layout(FLOOR_SAMPLED_TEXTURES + 1, "seventeen-texture layout");
    assert!(
        over.is_err(),
        "requesting Limits::default() must refuse 17 fragment-visible sampled \
         textures even though this adapter reports {}; without this refusal the \
         positive result would only be evidence about this GPU",
        gpu.adapter_reported
    );

    write_receipt(&gpu, observed, expected, &shader_sha256, &over.err());
}

/// A device below the floor cannot exist in collide-o-scope: the production
/// request would have failed. This records that the floor is a hard admission
/// gate rather than a hint.
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_the_device_floor_is_a_hard_admission_gate() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("GPU adapter");

    let impossible = wgpu::Limits {
        max_sampled_textures_per_shader_stage: u32::MAX,
        ..Default::default()
    };
    let refused = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Above-capability probe"),
        required_features: wgpu::Features::empty(),
        required_limits: impossible,
        ..Default::default()
    }));
    assert!(
        refused.is_err(),
        "request_device must refuse limits the adapter cannot satisfy; this is \
         the mechanism that makes Limits::default() a guaranteed floor rather \
         than a best effort"
    );
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

/// Resolve one git revision fact for the receipt.
///
/// The provenance fields were literals, so a receipt regenerated on another
/// machine kept claiming it had been measured at the original probe commit.
/// That is exactly backwards for an artifact whose whole purpose is to say
/// which hardware proved the claim and when. A tree with no git metadata - an
/// unpacked archive, a vendored copy - honestly reports `unknown` rather than
/// inheriting the previous run's answer.
fn measured_revision(flag: &str, target: &str) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", flag, target])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn write_receipt(
    gpu: &FloorHarness,
    observed: f32,
    expected: f32,
    shader_sha256: &str,
    refusal: &Option<wgpu::Error>,
) {
    let info = &gpu.adapter_info;
    let receipt = format!(
        r#"{{
  "receipt": "gpu_eight_sampled_textures_in_one_pass_are_portable_under_the_device_floor",
  "question": "Is one pass that simultaneously samples eight textures portable under the collide-o-scope device floor?",
  "verdict": "PASS",
  "claim_first_proven": {{
    "commit": "4866d34",
    "branch": "probe/s2-eight-texture-floor"
  }},
  "measured_at": {{
    "commit": "{measured_commit}",
    "branch": "{measured_branch}"
  }},
  "adapter": {{
    "name": "{name}",
    "backend": "{backend}",
    "device_type": "{device_type:?}",
    "driver": "{driver}",
    "driver_info": "{driver_info}",
    "vendor": {vendor},
    "device": {device}
  }},
  "floor": {{
    "source": "src/renderer/state.rs:2892",
    "required_features": "wgpu::Features::empty()",
    "required_limits": "wgpu::Limits::default()",
    "wgpu_version": "29.0.3",
    "max_sampled_textures_per_shader_stage": {floor},
    "enforced_on_device": {enforced},
    "adapter_raw_capability": {raw}
  }},
  "evidence": {{
    "textures_sampled_in_one_pass": {count},
    "bind_groups_used": 1,
    "shader_sha256": "{shader_sha256}",
    "naga_validation": "passed",
    "pipeline_creation": "passed",
    "expected_fingerprint": {expected},
    "observed_fingerprint": {observed},
    "fingerprint_exact_in_binary16": true,
    "floor_maximum_16_layout": "accepted",
    "over_floor_17_layout": "refused",
    "refusal_detail": "{refusal}"
  }},
  "interpretation": [
    "The device was created with exactly the production required_limits, so wgpu validated every operation against the floor's 16-texture budget rather than this adapter's raw capability.",
    "The 17-texture refusal on the same device demonstrates the cap was actively enforced, which is what makes the 8-texture success generalize.",
    "request_device refuses any adapter below the floor, so no adapter that runs collide-o-scope can offer fewer than 16 sampled textures per fragment stage.",
    "Eight is therefore portable with eight textures of headroom. The blocking constant is MAX_SAMPLED_TEXTURES_PER_PASS = 3 in src/visual_rack.rs:43, which is self-imposed policy, not a device capability."
  ],
  "scope_limits": [
    "This receipt proves capability under the floor. It does not measure performance, bandwidth, or cache behaviour of an eight-tap pass.",
    "One adapter and one backend were exercised; the portability claim rests on the enforced-cap argument, not on backend coverage.",
    "No product module, shader, or admission constant was modified.",
    "This file is regenerated by the probe that produced it. A changed receipt is a new measurement on new hardware, not drift: commit it. `claim_first_proven` is the frozen original proof and never moves; `measured_at` records the commit and branch this run actually measured."
  ]
}}
"#,
        measured_commit = measured_revision("--short", "HEAD"),
        measured_branch = measured_revision("--abbrev-ref", "HEAD"),
        name = info.name,
        backend = info.backend,
        device_type = info.device_type,
        driver = info.driver,
        driver_info = info.driver_info,
        vendor = info.vendor,
        device = info.device,
        floor = FLOOR_SAMPLED_TEXTURES,
        enforced = gpu.enforced_sampled_textures(),
        raw = gpu.adapter_reported,
        count = PROBE_TEXTURE_COUNT,
        shader_sha256 = shader_sha256,
        expected = expected,
        observed = observed,
        refusal = refusal
            .as_ref()
            .map(|error| {
                error
                    .to_string()
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default()
            .replace('"', "'")
            .replace('\\', "/"),
    );

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/s2-eight-texture-floor-receipt.json"
    );
    let mut file = std::fs::File::create(path).expect("receipt file");
    file.write_all(receipt.as_bytes()).expect("receipt write");

    println!("\n===== S2 GPU RECEIPT =====\n{receipt}\nwritten to: {path}\n");
}

// ---------------------------------------------------------------------------
// binary16 helpers (mirrors src/renderer/rack.rs:2367)
// ---------------------------------------------------------------------------

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let mantissa = (value & 0x03ff) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut exponent = -1i32;
        let mut mantissa = mantissa;
        while mantissa & 0x0400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        let mantissa = mantissa & 0x03ff;
        return f32::from_bits(
            sign | (((exponent + 127 - 15 + 1) as u32) << 23) | (mantissa << 13),
        );
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}
