// B11 deferred diagnostic sources. Both entry points write a linear
// Rgba8Unorm target so diagnostic bytes survive without the colour transfer
// performed by the finished-program sRGB reduction target.

const MONITOR_WIDTH: u32 = 128u;
const MONITOR_HEIGHT: u32 = 72u;

@group(0) @binding(0) var rgba_source: texture_2d<f32>;
@group(0) @binding(1) var motion_vectors: texture_2d<f32>;
@group(0) @binding(2) var motion_gates: texture_2d<f32>;

struct MotionMonitorUniforms {
    grid: vec2<u32>,
    max_uv_per_second: f32,
    _pad: u32,
};

@group(0) @binding(3) var<uniform> motion: MotionMonitorUniforms;

struct ExactVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_exact(@builtin(vertex_index) vertex_index: u32) -> ExactVertexOutput {
    var output: ExactVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    return output;
}

fn monitor_cell(position: vec4<f32>) -> vec2<u32> {
    return min(
        vec2<u32>(position.xy),
        vec2<u32>(MONITOR_WIDTH - 1u, MONITOR_HEIGHT - 1u),
    );
}

fn scaled_texel(cell: vec2<u32>, dimensions: vec2<u32>) -> vec2<i32> {
    let safe_dimensions = max(dimensions, vec2<u32>(1u));
    return vec2<i32>(min(
        vec2<u32>(
            cell.x * safe_dimensions.x / MONITOR_WIDTH,
            cell.y * safe_dimensions.y / MONITOR_HEIGHT,
        ),
        safe_dimensions - vec2<u32>(1u),
    ));
}

@fragment
fn fs_rgba(input: ExactVertexOutput) -> @location(0) vec4<f32> {
    let cell = monitor_cell(input.position);
    return textureLoad(rgba_source, scaled_texel(cell, textureDimensions(rgba_source)), 0);
}

fn finite_or_zero(value: vec2<f32>) -> vec2<f32> {
    // The pinned WGSL core has no portable isNan/isInf. An all-ones IEEE-754
    // exponent identifies both, expression-for-expression with the engine's
    // finite-or-zero law before clamping.
    let exponent = bitcast<vec2<u32>>(value) & vec2<u32>(0x7f800000u);
    return select(value, vec2<f32>(0.0), exponent == vec2<u32>(0x7f800000u));
}

@fragment
fn fs_motion(input: ExactVertexOutput) -> @location(0) vec4<f32> {
    let cell = monitor_cell(input.position);
    let texel = scaled_texel(cell, motion.grid);
    let velocity = finite_or_zero(textureLoad(motion_vectors, texel, 0).xy);
    let gate = clamp(
        finite_or_zero(textureLoad(motion_gates, texel, 0).xy),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let red = clamp(
        (velocity.x / motion.max_uv_per_second + 1.0) * 0.5,
        0.0,
        1.0,
    );
    let green = clamp(
        (1.0 - velocity.y / motion.max_uv_per_second) * 0.5,
        0.0,
        1.0,
    );
    return vec4<f32>(red, green, gate.x * gate.y, 1.0);
}
