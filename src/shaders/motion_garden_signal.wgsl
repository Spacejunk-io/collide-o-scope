// Low-resolution routed Refresh Garden signal.
//
// The vector field and its confidence/visibility gate are the exact staged
// parity selected by MotionMemoryStage for this frame. The output is a bounded
// scalar R8 texture consumed by the post-temporal Garden pass.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var vectors: texture_2d<f32>;
@group(0) @binding(1) var gates: texture_2d<f32>;
@group(0) @binding(2) var nearest_sampler: sampler;

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let velocity = textureSampleLevel(vectors, nearest_sampler, input.uv, 0.0).xy;
    let gate = textureSampleLevel(gates, nearest_sampler, input.uv, 0.0).xy;
    let signal = clamp(length(velocity) * gate.x * gate.y, 0.0, 1.0);
    return vec4<f32>(signal, 0.0, 0.0, 1.0);
}
