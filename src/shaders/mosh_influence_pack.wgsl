// Writes only target alpha. RGB arrives by a byte-exact texture copy from the
// opaque audience image, so enabling layer sends cannot perturb clean colour.

@group(0) @binding(0) var influence_mask: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    var uvs = array<vec2f, 3>(
        vec2f(0.0, 2.0),
        vec2f(0.0, 0.0),
        vec2f(2.0, 0.0),
    );
    var output: VertexOutput;
    output.position = vec4f(positions[index], 0.0, 1.0);
    output.uv = uvs[index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, textureSample(influence_mask, linear_sampler, input.uv).r);
}
