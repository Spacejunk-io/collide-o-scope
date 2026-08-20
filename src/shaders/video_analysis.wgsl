// B10 video content-analysis reduction: the program's pre-blackout opaque
// audience image reduced to the fixed 32x18 analysis grid BENDR's law is
// tuned to. Each output cell is the mean of a fixed 4x4 sub-grid of bilinear
// taps, sampled at explicit level 0 through a filtering sampler (the
// opaque-resolve precedent at this seam) and filtered in linear light; the
// sRGB target re-encodes on store. The CPU reference is
// `modulation::reduce_video_analysis_grid`, followed expression for
// expression.

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct ReduceVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_reduce(@builtin(vertex_index) vertex_index: u32) -> ReduceVertexOutput {
    // The established fullscreen triangle from `vertex_index`.
    var output: ReduceVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    return output;
}

@fragment
fn fs_reduce(input: ReduceVertexOutput) -> @location(0) vec4<f32> {
    let cell = floor(input.position.xy);
    var sum = vec4<f32>(0.0);
    for (var tap_y = 0u; tap_y < 4u; tap_y = tap_y + 1u) {
        for (var tap_x = 0u; tap_x < 4u; tap_x = tap_x + 1u) {
            let u = (cell.x + (f32(tap_x) + 0.5) / 4.0) / 32.0;
            let v = (cell.y + (f32(tap_y) + 0.5) / 4.0) / 18.0;
            sum = sum + textureSampleLevel(source_texture, source_sampler, vec2<f32>(u, v), 0.0);
        }
    }
    return sum / 16.0;
}
