// D2 evaluation-only photosensitivity-risk measurement kernel.
//
// One invocation owns one cell of the fixed 64x36 lattice and performs the
// same 4x4 integer-coordinate textureLoad pattern as the CPU reference. The
// source is read-only. The only copied result is eight aggregate u32 counters;
// no pixel or authored metadata can enter the readback buffer.

const GRID_WIDTH: u32 = 64u;
const GRID_HEIGHT: u32 = 36u;
const TAPS_PER_AXIS: u32 = 4u;
const TAPS_PER_CELL: u32 = 16u;
const Q_MAX: f32 = 65535.0;

struct CellHistory {
    // Linear-light red, green, blue, then Rec.709 luma, all Q0.16.
    rgb_luma: vec4<u32>,
    direction: i32,
    initialized: u32,
    _padding: vec2<u32>,
};

struct CompactCounters {
    sampled_cells: atomic<u32>,
    initialized_cells: atomic<u32>,
    affected_cells: atomic<u32>,
    reversal_cells: atomic<u32>,
    red_transition_cells: atomic<u32>,
    luma_delta_sum_q: atomic<u32>,
    color_delta_sum_q: atomic<u32>,
    reserved: atomic<u32>,
};

struct ReductionPolicy {
    transition_threshold_q: u32,
    red_saturation_q: u32,
    red_dominance_q: u32,
    reserved: u32,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> history: array<CellHistory>;
@group(0) @binding(2) var<storage, read_write> counters: CompactCounters;
@group(0) @binding(3) var<uniform> policy: ReductionPolicy;

fn sample_coordinate(
    cell: u32,
    tap: u32,
    grid_extent: u32,
    raster_extent: u32,
) -> u32 {
    let subdivision = cell * TAPS_PER_AXIS + tap;
    let numerator = (subdivision * 2u + 1u) * raster_extent;
    let denominator = grid_extent * TAPS_PER_AXIS * 2u;
    return min(numerator / denominator, raster_extent - 1u);
}

fn quantize_linear(linear: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(floor(clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0)) * Q_MAX + 0.5));
}

fn luma_q(rgb: vec3<u32>) -> u32 {
    // Rec.709 integer weights sum to exactly 65,536.
    return (rgb.r * 13933u + rgb.g * 46871u + rgb.b * 4732u + 32768u) / 65536u;
}

fn abs_diff(left: u32, right: u32) -> u32 {
    return select(right - left, left - right, left >= right);
}

fn red_saturated(sample: vec4<u32>) -> bool {
    return sample.r >= policy.red_saturation_q
        && sample.r >= sample.g + policy.red_dominance_q
        && sample.r >= sample.b + policy.red_dominance_q;
}

@compute @workgroup_size(64, 1, 1)
fn reduce(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let cell_index = invocation.x;
    if (cell_index >= GRID_WIDTH * GRID_HEIGHT) {
        return;
    }

    let cell_x = cell_index % GRID_WIDTH;
    let cell_y = cell_index / GRID_WIDTH;
    let dimensions = textureDimensions(source_texture, 0);
    var rgb_sum = vec3<u32>(0u);
    for (var tap_y = 0u; tap_y < TAPS_PER_AXIS; tap_y = tap_y + 1u) {
        for (var tap_x = 0u; tap_x < TAPS_PER_AXIS; tap_x = tap_x + 1u) {
            let x = sample_coordinate(cell_x, tap_x, GRID_WIDTH, dimensions.x);
            let y = sample_coordinate(cell_y, tap_y, GRID_HEIGHT, dimensions.y);
            let linear = textureLoad(source_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
            rgb_sum = rgb_sum + quantize_linear(linear);
        }
    }
    let rgb = (rgb_sum + vec3<u32>(TAPS_PER_CELL / 2u)) / TAPS_PER_CELL;
    let current = vec4<u32>(rgb, luma_q(rgb));
    let previous = history[cell_index];
    atomicAdd(&counters.sampled_cells, 1u);

    if (previous.initialized == 0u) {
        history[cell_index].rgb_luma = current;
        history[cell_index].direction = 0;
        history[cell_index].initialized = 1u;
        return;
    }

    atomicAdd(&counters.initialized_cells, 1u);
    let luma_delta = abs_diff(current.a, previous.rgb_luma.a);
    let color_delta = max(
        abs_diff(current.r, previous.rgb_luma.r),
        max(
            abs_diff(current.g, previous.rgb_luma.g),
            abs_diff(current.b, previous.rgb_luma.b),
        ),
    );
    atomicAdd(&counters.luma_delta_sum_q, luma_delta);
    atomicAdd(&counters.color_delta_sum_q, color_delta);

    let affected = max(luma_delta, color_delta) >= policy.transition_threshold_q;
    if (affected) {
        atomicAdd(&counters.affected_cells, 1u);
        if (red_saturated(current) || red_saturated(previous.rgb_luma)) {
            atomicAdd(&counters.red_transition_cells, 1u);
        }
    }

    if (luma_delta >= policy.transition_threshold_q) {
        var direction = 0;
        if (current.a > previous.rgb_luma.a) {
            direction = 1;
        } else if (current.a < previous.rgb_luma.a) {
            direction = -1;
        }
        if (previous.direction != 0 && direction != 0 && direction != previous.direction) {
            atomicAdd(&counters.reversal_cells, 1u);
        }
        if (direction != 0) {
            history[cell_index].direction = direction;
        }
    }
    history[cell_index].rgb_luma = current;
}
