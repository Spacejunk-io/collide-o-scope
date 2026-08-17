struct LatticeUniforms {
    field_size: vec2<u32>,
    source_size: vec2<u32>,
    search_radius: u32,
    update_hz: u32,
    algorithm_version: u32,
    _reserved: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct LatticeOutput {
    @location(0) velocity: vec2<f32>,
    @location(1) gate: vec2<f32>,
};

@group(0) @binding(0) var current_luma: texture_2d<f32>;
@group(0) @binding(1) var previous_luma: texture_2d<f32>;
@group(0) @binding(2) var nearest_sampler: sampler;
@group(0) @binding(3) var<uniform> uniforms: LatticeUniforms;

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

fn candidate_cost(uv: vec2<f32>, offset: vec2<i32>) -> f32 {
    let dimensions = vec2<f32>(uniforms.field_size);
    let displaced = uv + vec2<f32>(offset) / dimensions;
    if (any(displaced < vec2<f32>(0.0)) || any(displaced > vec2<f32>(1.0))) {
        return 1.0e6;
    }
    // A deterministic cross-shaped SAD preserves stable tie ordering while
    // remaining bounded at the fixed r2/r4/r8 quality tiers.
    let step = 1.0 / dimensions;
    var cost = 0.0;
    for (var axis = 0; axis < 5; axis = axis + 1) {
        var tap = vec2<f32>(0.0);
        if (axis == 1) { tap = vec2<f32>(step.x, 0.0); }
        if (axis == 2) { tap = vec2<f32>(-step.x, 0.0); }
        if (axis == 3) { tap = vec2<f32>(0.0, step.y); }
        if (axis == 4) { tap = vec2<f32>(0.0, -step.y); }
        let current = textureSample(current_luma, nearest_sampler, uv + tap).r;
        let previous = textureSample(previous_luma, nearest_sampler, displaced + tap).r;
        cost = cost + abs(current - previous);
    }
    return cost;
}

@fragment
fn fs_main(input: VertexOutput) -> LatticeOutput {
    var best_offset = vec2<i32>(0);
    var best_cost = 1.0e20;
    var second_cost = 1.0e20;
    let radius = i32(min(uniforms.search_radius, 8u));
    // Lexicographic candidate order is part of the deterministic algorithm.
    for (var y = -8; y <= 8; y = y + 1) {
        for (var x = -8; x <= 8; x = x + 1) {
            if (abs(x) <= radius && abs(y) <= radius) {
                let cost = candidate_cost(input.uv, vec2<i32>(x, y));
                if (cost < best_cost) {
                    second_cost = best_cost;
                    best_cost = cost;
                    best_offset = vec2<i32>(x, y);
                } else if (cost < second_cost) {
                    second_cost = cost;
                }
            }
        }
    }
    let source = max(vec2<f32>(uniforms.source_size), vec2<f32>(1.0));
    let block_size = source / max(vec2<f32>(uniforms.field_size), vec2<f32>(1.0));
    let velocity = vec2<f32>(best_offset) * block_size / source * f32(uniforms.update_hz);
    let separation = max(second_cost - best_cost, 0.0);
    let confidence = clamp(separation / max(second_cost, 1.0 / 255.0), 0.0, 1.0);
    var output: LatticeOutput;
    output.velocity = clamp(velocity, vec2<f32>(-64.0), vec2<f32>(64.0));
    output.gate = vec2<f32>(confidence, 1.0);
    return output;
}
