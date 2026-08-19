// B2 procedural motion fields. This shader is the GPU twin of
// `motion::procedural_field_sample`, expression for expression; the CPU
// reference is the law and this file follows it. Field-kind codes are the
// permanent `ProceduralFieldKind` codes: 0 curl, 1 radial, 2 spiral,
// 3 contour, 4 chroma, 5 weave.
//
// Derived from ideas in BENDR (c) 2026 Steve Blythe (MIT); the flow-field
// vocabulary follows its flow stage, rewritten for wgpu/WGSL.

struct ProceduralUniforms {
    field_size: vec2<u32>,
    kind: u32,
    _pad0: u32,
    // x time_seconds, y scale, z rate, w unused.
    values: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct ProceduralOutput {
    @location(0) velocity: vec2<f32>,
    @location(1) gate: vec2<f32>,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: ProceduralUniforms;

const TAU: f32 = 6.2831853071795864769;
const MAX_SPEED: f32 = 8.0;
const MAX_VELOCITY: f32 = 64.0;
const FRAC_1_SQRT_2: f32 = 0.70710678118654752440;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = positions[vertex_index] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return output;
}

// Covered premultiplied bilinear observation of the recipient's image —
// the exact quantity `covered_source_linear` computes in motion_luma.wgsl,
// so hostile RGB hidden behind zero coverage steers nothing.
fn covered_image(uv: vec2<f32>) -> vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(source_texture));
    let coordinate = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2<i32>(1);
    let p00 = clamp(base, vec2<i32>(0), maximum);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), maximum);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), maximum);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), maximum);
    let s00 = textureLoad(source_texture, p00, 0);
    let s10 = textureLoad(source_texture, p10, 0);
    let s01 = textureLoad(source_texture, p01, 0);
    let s11 = textureLoad(source_texture, p11, 0);
    let c00 = vec4<f32>(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4<f32>(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4<f32>(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4<f32>(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

fn covered_luma(uv: vec2<f32>) -> f32 {
    let covered = covered_image(uv);
    return dot(covered.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(input: VertexOutput) -> ProceduralOutput {
    let time_seconds = uniforms.values.x;
    let scale = clamp(uniforms.values.y, 0.0, 1.0);
    let rate = clamp(uniforms.values.z, -2.0, 2.0);
    let freq = 1.0 + scale * 15.0;
    let phase = max(time_seconds, 0.0) * rate;
    let p = input.uv - vec2<f32>(0.5);
    var velocity = vec2<f32>(0.0);
    var confidence = 1.0;
    switch uniforms.kind {
        case 0u: {
            // Curl: v = (dpsi/dy, -dpsi/dx) of a three-octave sinusoidal
            // stream function — divergence-free by construction. The three
            // octaves are the frozen CURL_OCTAVES constants of the CPU law.
            var gradient = vec2<f32>(0.0);
            var weight = 1.0;
            var octaves = array<vec4<f32>, 3>(
                // xy wave vector, z phase speed, w phase offset.
                vec4<f32>(1.0, 0.618, 1.0, 0.0),
                vec4<f32>(-1.618, 1.0, -0.5, 0.25),
                vec4<f32>(0.786, -1.376, 0.25, 0.5),
            );
            for (var octave = 0; octave < 3; octave = octave + 1) {
                let entry = octaves[octave];
                let argument = TAU * (freq * dot(entry.xy, input.uv) + entry.w + entry.z * phase);
                let ring = weight * cos(argument);
                gradient = gradient + ring * entry.xy;
                weight = weight * 0.5;
            }
            velocity = MAX_SPEED * vec2<f32>(gradient.y, -gradient.x);
        }
        case 1u, 2u: {
            // Radial pulses outward; Spiral pitches the same ring 45 degrees.
            let r = length(p);
            if (r > 1.0e-6) {
                let outward = p / r;
                var direction = outward;
                if (uniforms.kind == 2u) {
                    direction = vec2<f32>(
                        (outward.x - outward.y) * FRAC_1_SQRT_2,
                        (outward.y + outward.x) * FRAC_1_SQRT_2,
                    );
                }
                let ring = cos(TAU * (freq * r - phase));
                velocity = MAX_SPEED * direction * ring;
            }
        }
        case 3u: {
            // Contour: flow along luma isolines — perpendicular to the
            // central-difference gradient measured one field cell out.
            let cell_step = vec2<f32>(1.0) / max(vec2<f32>(uniforms.field_size), vec2<f32>(1.0));
            let gx = (covered_luma(input.uv + vec2<f32>(cell_step.x, 0.0))
                - covered_luma(input.uv - vec2<f32>(cell_step.x, 0.0))) * 0.5;
            let gy = (covered_luma(input.uv + vec2<f32>(0.0, cell_step.y))
                - covered_luma(input.uv - vec2<f32>(0.0, cell_step.y))) * 0.5;
            let tangent = vec2<f32>(-gy, gx);
            let swing = cos(TAU * phase);
            velocity = MAX_SPEED * tangent * freq * swing;
            confidence = clamp(length(vec2<f32>(gx, gy)) * 8.0, 0.0, 1.0);
        }
        case 4u: {
            // Chroma: steer by the alpha-covered YIQ chroma pair, rotated by
            // the phase so rate spins the steering.
            let covered = covered_image(input.uv);
            let i = 0.596 * covered.r - 0.274 * covered.g - 0.322 * covered.b;
            let q = 0.211 * covered.r - 0.523 * covered.g + 0.312 * covered.b;
            let angle = TAU * phase;
            let steered = vec2<f32>(
                i * cos(angle) - q * sin(angle),
                i * sin(angle) + q * cos(angle),
            );
            velocity = MAX_SPEED * 2.0 * steered;
            confidence = clamp(length(vec2<f32>(i, q)) * 4.0, 0.0, 1.0);
        }
        default: {
            // Weave: orthogonal sinusoidal shear bands.
            velocity = vec2<f32>(
                MAX_SPEED * sin(TAU * (freq * input.uv.y + phase)),
                MAX_SPEED * 0.25 * sin(TAU * (freq * input.uv.x - phase)),
            );
        }
    }
    var output: ProceduralOutput;
    output.velocity = clamp(velocity, vec2<f32>(-MAX_VELOCITY), vec2<f32>(MAX_VELOCITY));
    output.gate = vec2<f32>(confidence, 1.0);
    return output;
}
