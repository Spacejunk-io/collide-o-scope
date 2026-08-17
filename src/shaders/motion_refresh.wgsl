struct RefreshUniforms {
    faraday_values: vec4<f32>,
    gate_values: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var current_texture: texture_2d<f32>;
@group(0) @binding(1) var advected_texture: texture_2d<f32>;
@group(0) @binding(2) var gate_texture: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var nearest_sampler: sampler;
@group(0) @binding(5) var<uniform> uniforms: RefreshUniforms;

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

fn premultiply_refresh(straight: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4<f32>(straight.rgb * alpha, alpha);
}

fn straight_from_refresh_premultiplied(value: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(value.a, 0.0, 1.0);
    if (alpha <= 0.000001) { return vec4<f32>(0.0); }
    return vec4<f32>(value.rgb / alpha, alpha);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let current = premultiply_refresh(textureSample(current_texture, linear_sampler, input.uv));
    let advected = premultiply_refresh(textureSample(advected_texture, linear_sampler, input.uv));
    let gate_sample = textureSample(gate_texture, nearest_sampler, input.uv).xy;
    let threshold = uniforms.gate_values.x;
    let softness = uniforms.gate_values.y;
    let confidence = select(
        select(0.0, 1.0, gate_sample.x >= threshold),
        smoothstep(threshold - softness, threshold + softness, gate_sample.x),
        softness > 0.000001,
    );
    let gate = confidence * mix(1.0, gate_sample.y, uniforms.faraday_values.w);
    let amount = uniforms.faraday_values.x * gate;
    let refresh = uniforms.faraday_values.y;
    let decay = uniforms.faraday_values.z;
    let memory = mix(advected * decay, current, refresh);
    return straight_from_refresh_premultiplied(mix(current, memory, amount));
}
