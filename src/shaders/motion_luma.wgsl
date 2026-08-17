struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

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

fn covered_source_linear(uv: vec2<f32>) -> vec4<f32> {
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) f32 {
    let covered = covered_source_linear(input.uv);
    return dot(covered.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}
