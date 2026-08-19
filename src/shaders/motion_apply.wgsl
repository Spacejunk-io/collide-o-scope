struct MotionSpatialSample {
    red_row_0: vec4<f32>,
    red_row_1: vec4<f32>,
    green_row_0: vec4<f32>,
    green_row_1: vec4<f32>,
    blue_row_0: vec4<f32>,
    blue_row_1: vec4<f32>,
};

struct MotionApplyUniforms {
    output_to_donor_row_0: vec4<f32>,
    output_to_donor_row_1: vec4<f32>,
    donor_to_recipient_row_0: vec4<f32>,
    donor_to_recipient_row_1: vec4<f32>,
    shutter_values: vec4<f32>,
    faraday_values: vec4<f32>,
    // x delta seconds, y frame-plan program time, z/w unused.
    frame_values: vec4<f32>,
    modes: vec4<u32>,
    // B2 flow shaping: x stretch, y edge_repel, z vector_trash, w trash
    // block size in output pixels. All-zero amounts are the exact prior
    // path: no extra texture operation and no velocity contribution.
    shaping_values: vec4<f32>,
    spatial_samples: array<MotionSpatialSample, 16>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var carrier_texture: texture_2d<f32>;
@group(0) @binding(1) var vector_texture: texture_2d<f32>;
@group(0) @binding(2) var gate_texture: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var nearest_sampler: sampler;
@group(0) @binding(5) var<uniform> uniforms: MotionApplyUniforms;

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

// The shared integer avalanche, byte-identical to effects.wgsl's
// cellular_avalanche and to motion.rs's motion_avalanche.
fn shaping_avalanche(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// One deterministic unit sample for a trash cell, epoch, and lane. The
// 24-bit mantissa path is the CPU reference's `flow_trash_hash`, expression
// for expression, under the same "MTRS" domain constant.
fn flow_trash_hash(cell: vec2<i32>, epoch: u32, lane: u32) -> f32 {
    let seed = shaping_avalanche(
        bitcast<u32>(cell.x)
            ^ (bitcast<u32>(cell.y) * 0x9e3779b9u)
            ^ (epoch * 0x85ebca6bu)
            ^ (lane * 0x27d4eb2fu)
            ^ 0x4d545253u,
    );
    return f32(seed & 0x00ffffffu) / 16777216.0;
}

fn covered_carrier_luma(uv: vec2<f32>) -> f32 {
    let sample = textureSample(carrier_texture, linear_sampler, uv);
    return dot(sample.rgb * clamp(sample.a, 0.0, 1.0), vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The B2 flow-shaping law over the gated sampled velocity: stretch, then
// repel, then trash, then the canonical clamp — the CPU reference
// `shape_flow_velocity`, expression for expression. Each sub-block guards on
// its own amount so an authored zero takes no extra texture operation.
fn shape_flow(velocity: vec2<f32>, uv: vec2<f32>) -> vec2<f32> {
    var v = velocity;
    let stretch = uniforms.shaping_values.x;
    if (stretch > 0.0) {
        let p = uv - vec2<f32>(0.5);
        let r = length(p);
        if (r > 1.0e-6) {
            v = v + p / r * length(v) * stretch;
        }
    }
    let edge_repel = uniforms.shaping_values.y;
    if (edge_repel > 0.0) {
        let step = vec2<f32>(1.0) / max(vec2<f32>(textureDimensions(carrier_texture)), vec2<f32>(1.0));
        let gx = (covered_carrier_luma(uv + vec2<f32>(step.x, 0.0))
            - covered_carrier_luma(uv - vec2<f32>(step.x, 0.0))) * 0.5;
        let gy = (covered_carrier_luma(uv + vec2<f32>(0.0, step.y))
            - covered_carrier_luma(uv - vec2<f32>(0.0, step.y))) * 0.5;
        let g = vec2<f32>(gx, gy);
        let magnitude = length(g);
        if (magnitude > 1.0e-6) {
            let push = min(magnitude * 8.0, 1.0) * 8.0 * edge_repel;
            v = v - g / magnitude * push;
        }
    }
    let vector_trash = uniforms.shaping_values.z;
    if (vector_trash > 0.0) {
        let block = uniforms.shaping_values.w;
        let output_px = uv * vec2<f32>(textureDimensions(carrier_texture));
        let cell = vec2<i32>(floor(output_px / block));
        let epoch = u32(max(uniforms.frame_values.y, 0.0) * 8.0);
        if (flow_trash_hash(cell, epoch, 0u) < vector_trash) {
            let garbage = vec2<f32>(
                flow_trash_hash(cell, epoch, 1u) * 2.0 - 1.0,
                flow_trash_hash(cell, epoch, 2u) * 2.0 - 1.0,
            );
            v = v + garbage * 16.0;
        }
    }
    return clamp(v, vec2<f32>(-64.0), vec2<f32>(64.0));
}

fn gate_weight(gate: vec2<f32>) -> f32 {
    let threshold = uniforms.faraday_values.y;
    let softness = uniforms.faraday_values.z;
    var confidence = select(
        select(0.0, 1.0, gate.x >= threshold),
        smoothstep(threshold - softness, threshold + softness, gate.x),
        softness > 0.000001,
    );
    return confidence * mix(1.0, gate.y, uniforms.faraday_values.w);
}

fn trajectory(value: f32) -> f32 {
    let curvature = uniforms.shutter_values.z;
    return value + curvature * value * abs(value) * 0.5;
}

fn mapped_uv(uv: vec2<f32>, row_0: vec4<f32>, row_1: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        dot(row_0.xyz, vec3<f32>(uv, 1.0)),
        dot(row_1.xyz, vec3<f32>(uv, 1.0)),
    );
}

fn straight_from_premultiplied_filter(value: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(value.a, 0.0, 1.0);
    if (alpha <= 0.000001) { return vec4<f32>(0.0); }
    return vec4<f32>(value.rgb / alpha, alpha);
}

fn sample_carrier_premultiplied_linear(uv: vec2<f32>) -> vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(carrier_texture));
    let coordinate = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2<i32>(1);
    let p00 = clamp(base, vec2<i32>(0), maximum);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), maximum);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), maximum);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), maximum);
    let s00 = textureLoad(carrier_texture, p00, 0);
    let s10 = textureLoad(carrier_texture, p10, 0);
    let s01 = textureLoad(carrier_texture, p01, 0);
    let s11 = textureLoad(carrier_texture, p11, 0);
    let c00 = vec4<f32>(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4<f32>(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4<f32>(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4<f32>(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    return straight_from_premultiplied_filter(
        mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y),
    );
}

fn trajectory_sample(
    uv: vec2<f32>,
    velocity: vec2<f32>,
    value: f32,
    row_0: vec4<f32>,
    row_1: vec4<f32>,
) -> vec4<f32> {
    // A shutter angle owns its authored exposure. Faraday-only induction uses
    // its bounded amount as a one-frame advection scale instead of collapsing
    // to a zero-offset sample when the exact shutter fast path is selected.
    let exposure = select(
        uniforms.faraday_values.x,
        uniforms.shutter_values.x / 360.0,
        uniforms.shutter_values.x > 0.0,
    );
    let offset = velocity * uniforms.frame_values.x * exposure * trajectory(value);
    let transformed_uv = mapped_uv(uv, row_0, row_1);
    let coordinate = clamp(transformed_uv - offset, vec2<f32>(0.0), vec2<f32>(1.0));
    return sample_carrier_premultiplied_linear(coordinate);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let exact_zero = uniforms.modes.x == 0u;
    if (exact_zero) {
        // Exact-zero preserves the existing Advanced no-op command path.
        return textureSample(carrier_texture, linear_sampler, input.uv);
    }
    var velocity = vec2<f32>(0.0);
    if (uniforms.modes.w != 0u) {
        let donor_uv = vec2<f32>(
            dot(uniforms.output_to_donor_row_0.xyz, vec3<f32>(input.uv, 1.0)),
            dot(uniforms.output_to_donor_row_1.xyz, vec3<f32>(input.uv, 1.0)),
        );
        if (all(donor_uv >= vec2<f32>(0.0)) && all(donor_uv <= vec2<f32>(1.0))) {
            let donor_velocity = textureSample(vector_texture, nearest_sampler, donor_uv).xy;
            let mapped_velocity = vec2<f32>(
                dot(uniforms.donor_to_recipient_row_0.xy, donor_velocity),
                dot(uniforms.donor_to_recipient_row_1.xy, donor_velocity),
            );
            let gate = gate_weight(textureSample(gate_texture, nearest_sampler, donor_uv).xy);
            velocity = mapped_velocity * gate;
        }
        // Shaping modifies an APPLIED field: it runs only under a valid
        // field, and only when an amount is authored, so all-zero shaping is
        // byte-exact with the unshaped path — no clamp, no texture op.
        let shaping = uniforms.shaping_values;
        if (shaping.x > 0.0 || shaping.y > 0.0 || shaping.z > 0.0) {
            velocity = shape_flow(velocity, input.uv);
        }
    }
    let sample_count = clamp(uniforms.modes.y, 1u, 16u);
    let phase = uniforms.shutter_values.y;
    let chromatic_lag = uniforms.shutter_values.w;
    var accumulated = vec4<f32>(0.0);
    for (var index = 0u; index < 16u; index = index + 1u) {
        if (index < sample_count) {
            let denominator = f32(max(sample_count - 1u, 1u));
            let time = f32(index) / denominator - 0.5 + phase * 0.5;
            let spatial = uniforms.spatial_samples[index];
            if (chromatic_lag > 0.0) {
                let lag = chromatic_lag / denominator;
                let red = trajectory_sample(
                    input.uv,
                    velocity,
                    time - lag,
                    spatial.red_row_0,
                    spatial.red_row_1,
                );
                let green = trajectory_sample(
                    input.uv,
                    velocity,
                    time,
                    spatial.green_row_0,
                    spatial.green_row_1,
                );
                let blue = trajectory_sample(
                    input.uv,
                    velocity,
                    time + lag,
                    spatial.blue_row_0,
                    spatial.blue_row_1,
                );
                let alpha = (red.a + green.a + blue.a) / 3.0;
                let premultiplied = vec3<f32>(
                    red.r * red.a,
                    green.g * green.a,
                    blue.b * blue.a,
                );
                accumulated = accumulated + vec4<f32>(premultiplied, alpha);
            } else {
                let sample = trajectory_sample(
                    input.uv,
                    velocity,
                    time,
                    spatial.green_row_0,
                    spatial.green_row_1,
                );
                accumulated = accumulated + vec4<f32>(
                    sample.rgb * clamp(sample.a, 0.0, 1.0),
                    clamp(sample.a, 0.0, 1.0),
                );
            }
        }
    }
    return straight_from_premultiplied_filter(accumulated / f32(sample_count));
}
