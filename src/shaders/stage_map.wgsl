// Post-creative StageMap presentation. This shader only samples the completed
// Program image and writes dedicated physical-endpoint targets.

const ROUTE_PROGRAM: u32 = 0u;
const ROUTE_BLACKOUT: u32 = 1u;
const ROUTE_SMPTE: u32 = 2u;
const ROUTE_GRID: u32 = 3u;

const MASK_NONE: u32 = 0u;
const MASK_EDGE_FEATHER: u32 = 1u;
const MASK_POLYGON: u32 = 2u;

struct StageUniforms {
    homography_0: vec4f,
    homography_1: vec4f,
    homography_2: vec4f,
    calibration: vec4f,
    gain_mask: vec4f,
    black_invert: vec4f,
    feather: vec4f,
    bounds: vec4f,
    mask_points_0: vec4f,
    mask_points_1: vec4f,
    mask_points_2: vec4f,
    mask_points_3: vec4f,
    modes: vec4u,
    surface: vec4u,
    reserved_0: vec4u,
    reserved_1: vec4u,
};

@group(0) @binding(0) var program_texture: texture_2d<f32>;
@group(0) @binding(1) var program_sampler: sampler;
@group(1) @binding(0) var<uniform> stage: StageUniforms;

struct SliceVertexInput {
    @location(0) source_uv: vec2f,
    @location(1) output_uv: vec2f,
};

struct StageVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) source_uv: vec2f,
    @location(1) output_uv: vec2f,
};

@vertex
fn vs_slice(input: SliceVertexInput) -> StageVertexOutput {
    var output: StageVertexOutput;
    output.position = vec4f(
        input.output_uv.x * 2.0 - 1.0,
        1.0 - input.output_uv.y * 2.0,
        0.0,
        1.0,
    );
    output.source_uv = input.source_uv;
    output.output_uv = input.output_uv;
    return output;
}

@vertex
fn vs_surface(@builtin(vertex_index) index: u32) -> StageVertexOutput {
    let x = f32(i32(index & 1u)) * 4.0 - 1.0;
    let y = f32(i32(index >> 1u)) * 4.0 - 1.0;
    let uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    var output: StageVertexOutput;
    output.position = vec4f(x, y, 0.0, 1.0);
    output.source_uv = uv;
    output.output_uv = uv;
    return output;
}

fn mask_point(index: u32) -> vec2f {
    switch index {
        case 0u: { return stage.mask_points_0.xy; }
        case 1u: { return stage.mask_points_0.zw; }
        case 2u: { return stage.mask_points_1.xy; }
        case 3u: { return stage.mask_points_1.zw; }
        case 4u: { return stage.mask_points_2.xy; }
        case 5u: { return stage.mask_points_2.zw; }
        case 6u: { return stage.mask_points_3.xy; }
        default: { return stage.mask_points_3.zw; }
    }
}

fn soft_edge(distance: f32, softness: f32) -> f32 {
    if softness <= 0.0 {
        return select(0.0, 1.0, distance >= 0.0);
    }
    return smoothstep(0.0, softness, distance);
}

fn mask_coverage(uv: vec2f) -> f32 {
    let kind = stage.modes.y;
    if kind == MASK_NONE {
        return 1.0;
    }
    if kind == MASK_EDGE_FEATHER {
        var coverage = 1.0;
        coverage *= soft_edge(uv.x - stage.bounds.x, stage.feather.x);
        coverage *= soft_edge(uv.y - stage.bounds.y, stage.feather.y);
        coverage *= soft_edge(stage.bounds.z - uv.x, stage.feather.z);
        coverage *= soft_edge(stage.bounds.w - uv.y, stage.feather.w);
        return coverage;
    }

    let count = stage.modes.z;
    var minimum_distance = 2.0;
    for (var index = 0u; index < 8u; index += 1u) {
        if index < count {
            let next = select(index + 1u, 0u, index + 1u == count);
            let a = mask_point(index);
            let b = mask_point(next);
            let edge = b - a;
            let signed_distance = (edge.x * (uv.y - a.y) - edge.y * (uv.x - a.x))
                / max(length(edge), 1.0e-6);
            minimum_distance = min(minimum_distance, signed_distance);
        }
    }
    var coverage = soft_edge(minimum_distance, stage.feather.x);
    if stage.black_invert.w > 0.5 {
        coverage = 1.0 - coverage;
    }
    return coverage;
}

fn projective_uv(output_uv: vec2f) -> vec2f {
    let x = output_uv.x;
    let y = output_uv.y;
    let denominator = stage.homography_2.x * x
        + stage.homography_2.y * y
        + stage.homography_2.z;
    return vec2f(
        (stage.homography_0.x * x + stage.homography_0.y * y + stage.homography_0.z)
            / denominator,
        (stage.homography_1.x * x + stage.homography_1.y * y + stage.homography_1.z)
            / denominator,
    );
}

fn calibrate(sample: vec4f, coverage: f32) -> vec4f {
    let source_alpha = clamp(sample.a, 0.0, 1.0);
    let straight = select(vec3f(0.0), sample.rgb / max(source_alpha, 1.0e-6), source_alpha > 0.0);
    var color = max((straight - vec3f(0.5)) * stage.calibration.z + vec3f(0.5), vec3f(0.0));
    color *= stage.calibration.y;
    color = pow(color, vec3f(1.0 / stage.calibration.w));
    color = clamp(color * stage.gain_mask.xyz + stage.black_invert.xyz, vec3f(0.0), vec3f(1.0));
    let alpha = source_alpha * stage.calibration.x * coverage;
    return vec4f(color * alpha, alpha);
}

@fragment
fn fs_slice(input: StageVertexOutput) -> @location(0) vec4f {
    var source_uv = input.source_uv;
    if stage.modes.x != 0u {
        source_uv = projective_uv(input.output_uv);
    }
    let sample = textureSample(program_texture, program_sampler, source_uv);
    return calibrate(sample, mask_coverage(input.output_uv));
}

fn identifier_overlay(uv: vec2f) -> vec4f {
    if stage.surface.y == 0u {
        return vec4f(0.0);
    }
    let hash = stage.surface.z;
    let red = 0.25 + 0.75 * f32(hash & 255u) / 255.0;
    let green = 0.25 + 0.75 * f32((hash >> 8u) & 255u) / 255.0;
    let blue = 0.25 + 0.75 * f32((hash >> 16u) & 255u) / 255.0;
    let border = uv.x < 0.015 || uv.x > 0.985 || uv.y < 0.02 || uv.y > 0.98;
    let column = min(u32(floor(uv.x * 32.0)), 31u);
    let barcode_bit = (hash >> (column & 31u)) & 1u;
    let barcode = uv.y > 0.88 && uv.y < 0.96 && barcode_bit != 0u;
    if border || barcode {
        return vec4f(red, green, blue, 1.0);
    }
    return vec4f(0.0);
}

fn test_card(uv: vec2f, route: u32) -> vec4f {
    if route == ROUTE_SMPTE {
        let bars = array<vec3f, 7>(
            vec3f(0.75, 0.75, 0.75),
            vec3f(0.75, 0.75, 0.0),
            vec3f(0.0, 0.75, 0.75),
            vec3f(0.0, 0.75, 0.0),
            vec3f(0.75, 0.0, 0.75),
            vec3f(0.75, 0.0, 0.0),
            vec3f(0.0, 0.0, 0.75),
        );
        let index = min(u32(floor(uv.x * 7.0)), 6u);
        let lower = select(bars[index] * 0.18, vec3f(0.02), fract(uv.x * 14.0) > 0.5);
        return vec4f(select(lower, bars[index], uv.y < 0.75), 1.0);
    }
    if route == ROUTE_GRID {
        let major = min(fract(uv.x * 10.0), fract(uv.y * 10.0));
        let center = abs(uv.x - 0.5) < 0.004 || abs(uv.y - 0.5) < 0.004;
        let line = major < 0.025 || center;
        return vec4f(select(vec3f(0.03), vec3f(0.9), line), 1.0);
    }
    return vec4f(0.0);
}

@fragment
fn fs_surface(input: StageVertexOutput) -> @location(0) vec4f {
    let route = stage.surface.x;
    var base = test_card(input.output_uv, route);
    let overlay = identifier_overlay(input.output_uv);
    base = overlay + base * (1.0 - overlay.a);
    return base;
}

@fragment
fn fs_present(input: StageVertexOutput) -> @location(0) vec4f {
    return textureSample(program_texture, program_sampler, input.output_uv);
}
