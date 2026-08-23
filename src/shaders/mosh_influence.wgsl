// Program-space layer send matte for Codec Mosh. The RGB programme keeps its
// ordinary compositor; this one-channel companion follows post-local coverage
// and is evaluated only when at least one contributing layer authors a send
// below one.

struct InfluenceUniforms {
    send: f32,
    opacity: f32,
    blend_mode: u32,
    _pad: u32,
};

@group(0) @binding(0) var base_mask: texture_2d<f32>;
@group(0) @binding(1) var overlay_texture: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(1) @binding(0) var<uniform> influence: InfluenceUniforms;

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
    let base = textureSample(base_mask, linear_sampler, input.uv).r;
    let coverage = clamp(
        textureSample(overlay_texture, linear_sampler, input.uv).a
            * influence.opacity,
        0.0,
        1.0,
    );
    // BlendMode::AlphaCut is destination-out: the cutting shape owns no send
    // of its own and removes the prior control field with the same coverage.
    let next = select(
        mix(base, clamp(influence.send, 0.0, 1.0), coverage),
        base * (1.0 - coverage),
        influence.blend_mode == 14u,
    );
    return vec4f(clamp(next, 0.0, 1.0), 0.0, 0.0, 1.0);
}
