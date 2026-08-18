// Gesture-field etching: one ordered sample per pass over the twelve-byte
// canvas cell (an Rg16Float signed vector ping-pong pair plus an Rg8Unorm
// coverage/hold ping-pong pair).
//
// Every law here is the exact CPU reference in src/gesture_canvas.rs, written
// in the same order with the same spellings so the two can be compared rather
// than merely believed:
//
//   * `etch_falloff`             -> the `remaining * remaining` branch below
//   * `cell_retention_per_tick`  -> the `retention + (1 - retention) * hold`
//   * `GestureCanvasField::decay`-> the two `pow(..., decay_ticks)` factors
//   * `GestureCanvasField::etch` -> the blend toward `axis * strength`
//
// The carrier is read with `textureLoad` at the fragment's own integer
// coordinate, so no filter, address mode, or half-texel convention can drift
// between a cell and the reference cell it must equal.

struct GestureEtchUniforms {
    grid_size: vec2<u32>,
    // Decay ticks this pass applies. Only the first pass of a frame carries a
    // nonzero count; the budget clamp that produced it is a CPU law.
    decay_ticks: f32,
    // Authored per-tick retention for a wholly unheld cell.
    retention: f32,
    // x, y, pressure, radius.
    sample: vec4<f32>,
    // axis.x, axis.y, strength, active.
    axis: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct EtchOutput {
    @location(0) vector: vec2<f32>,
    @location(1) gate: vec2<f32>,
};

@group(0) @binding(0) var previous_vector: texture_2d<f32>;
@group(0) @binding(1) var previous_gate: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: GestureEtchUniforms;

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
fn fs_main(input: VertexOutput) -> EtchOutput {
    let coordinate = vec2<i32>(input.position.xy);
    var vector = textureLoad(previous_vector, coordinate, 0).rg;
    let previous = textureLoad(previous_gate, coordinate, 0).rg;
    var coverage = previous.r;
    var hold = previous.g;

    // Decay. Closed form, so a long gap is one operation rather than a loop;
    // the tick budget that bounds `decay_ticks` is enforced on the CPU.
    if (uniforms.decay_ticks > 0.0) {
        let authored = clamp(uniforms.retention, 0.0, 1.0);
        let cell_retention = authored + (1.0 - authored) * clamp(hold, 0.0, 1.0);
        let factor = pow(cell_retention, uniforms.decay_ticks);
        vector = vector * factor;
        coverage = coverage * factor;
        // `hold` decays at the authored rate too, so retention stays finite for
        // every cell and nothing is etched permanently.
        hold = hold * pow(authored, uniforms.decay_ticks);
    }

    // Etch. `active` is zero for an inert sample — no direction or no pressure
    // — and for a decay-only pass, and the whole branch is skipped rather than
    // being run with an invented direction.
    if (uniforms.axis.w > 0.5) {
        // The canvas position of this cell, in the same normalized space the
        // recorded event carries. `y` increases downward on both sides.
        let dimensions = max(vec2<f32>(uniforms.grid_size), vec2<f32>(1.0));
        let position = (vec2<f32>(coordinate) + vec2<f32>(0.5)) / dimensions;
        let delta = position - uniforms.sample.xy;
        let distance = sqrt(delta.x * delta.x + delta.y * delta.y);
        let radius = uniforms.sample.w;
        var falloff = 0.0;
        if (radius > 0.0 && distance < radius) {
            let remaining = 1.0 - distance / radius;
            falloff = remaining * remaining;
        }
        let pressure = clamp(uniforms.sample.z, 0.0, 1.0);
        let blend = falloff * pressure;
        if (blend > 0.0) {
            // A blend toward this sample's target, never an accumulation onto
            // it. That is what makes overlapping strokes compose in recorded
            // order instead of commuting.
            let etched = uniforms.axis.xy * uniforms.axis.z;
            vector = vector * (1.0 - blend) + etched * blend;
            coverage = coverage + blend * (1.0 - coverage);
            hold = hold + blend * (pressure - hold);
        }
    }

    var output: EtchOutput;
    output.vector = vector;
    output.gate = vec2<f32>(clamp(coverage, 0.0, 1.0), clamp(hold, 0.0, 1.0));
    return output;
}

// Publish the committed parity as the one routable donor image an ordinary
// composition image tap binds. This is the exact inverse of the already-frozen
// `displace_node` decode
//
//     vector = (premultiplied_rg - 0.5 * alpha) * 2
//
// written here as the exact spelling of `gesture_canvas::present_displace_donor`
// so the CPU reference and the device agree by comparison rather than by
// belief. It reuses this module's bind group layout unchanged — the uniform at
// binding 2 is simply unread — so presentation adds no bind group, no layout,
// and no sampler. `textureLoad` again, at the fragment's own integer
// coordinate, because the presented image is one-to-one with the canvas.
//
// Two exactness properties are arithmetic, not convention: an un-etched cell
// presents (0, 0, 0, 0) and decodes to exactly zero, and a zero-coverage cell
// decodes to exactly zero whatever vector it stores, because the premultiply
// uses the very alpha the decode subtracts.
@fragment
fn fs_present(input: VertexOutput) -> @location(0) vec4<f32> {
    let coordinate = vec2<i32>(input.position.xy);
    let vector = textureLoad(previous_vector, coordinate, 0).rg;
    let gate = textureLoad(previous_gate, coordinate, 0).rg;
    let alpha = clamp(gate.r, 0.0, 1.0);
    let straight = clamp(vector, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5 + vec2<f32>(0.5);
    // Blue is unused and is written as an explicit zero, never left undefined.
    return vec4<f32>(straight * alpha, 0.0, alpha);
}
