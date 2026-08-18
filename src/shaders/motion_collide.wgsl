// Field Collider v1 — two low-resolution passes over the recipient's grid.
//
// Pass 1 maps donor A's and donor B's vectors out of their own field spaces
// into recipient-local coordinates and packs both into ONE RGBA16Float pair
// surface, sampling two textures. Pass 2 samples that pair plus both gates —
// three textures — and writes the transactional derived RG16Float vector and
// RG8Unorm gate. Neither pass exceeds the unchanged three-sampled-texture
// ceiling, which is the whole reason the mapping is split out of pass 2.
//
// The two-pass split also means the derived field is already recipient-local
// and indexed in composition output UV, so the Faraday apply pass consumes it
// under identity transforms — there is no second vector mapping downstream.
//
// This shader is the dependent half of a pair: src/motion.rs owns the
// independent CPU reference (`collide_vectors`, `MotionBoundaryMode::resolve`,
// `clamp_motion_velocity`, `ColliderInputSample::validated`) and the fixtures
// measure this against that, expression for expression.

// Exactly 144 bytes: two 64-byte MotionTransformGpu records plus one 16-byte
// mode/status lane. The Rust twin carries a compile-time size assertion.
struct ColliderTransform {
    // Composition output UV -> this input's donor-local UV. Row 2 is the
    // translation column; it moves COORDINATES only.
    output_to_donor_row_0: vec4<f32>,
    output_to_donor_row_1: vec4<f32>,
    // linear(inverse(recipient) * donor). Only .xy is read, so translation can
    // never reach a vector.
    donor_to_recipient_row_0: vec4<f32>,
    donor_to_recipient_row_1: vec4<f32>,
};

struct ColliderUniforms {
    input_a: ColliderTransform,
    input_b: ColliderTransform,
    // x: FieldColliderMode code 0..4
    // y: MotionBoundaryMode code 0..3 (Transparent 0, Mirror 1, Wrap 2, Hold 3)
    // z: bit 0 = input A transform is finite and nonsingular and its committed
    //            parity is materialized; bit 1 = the same for input B
    // w: reserved, always zero
    modes: vec4<u32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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

// The canonical Motion velocity range. This is exactly the interval
// pack_velocity encodes and unpack_velocity recovers, so no mode can emit a
// velocity the frozen M4 field contract cannot represent.
const MOTION_MAX_UV_PER_SECOND: f32 = 64.0;
// Squared-magnitude floor below which a direction is not a direction.
const COLLIDER_EPSILON: f32 = 1e-12;
// The sentinel a removed lookup writes into the pair surface. It is outside the
// representable velocity range on purpose, so pass 2 can recognise a removed
// sample without a second surface, and `validated` below rejects it by the same
// range test the CPU reference applies.
const COLLIDER_INVALID: f32 = 1.0e30;

fn wrap_unit(value: f32) -> f32 {
    return clamp(value - floor(value), 0.0, 1.0);
}

fn mirror_unit(value: f32) -> f32 {
    let half_value = value / 2.0;
    let period = (half_value - floor(half_value)) * 2.0;
    var folded = period;
    if (period > 1.0) {
        folded = 2.0 - period;
    }
    return clamp(folded, 0.0, 1.0);
}

// Resolve one lookup coordinate under the authored boundary law. `removed` is
// set when the law rejects the coordinate; a non-finite coordinate is removed
// by EVERY law, because clamp/fract/triangle are all meaningless on NaN and
// inventing a coordinate there would fabricate a reading never taken.
struct BoundaryResult {
    uv: vec2<f32>,
    removed: bool,
};

fn resolve_boundary(boundary: u32, uv: vec2<f32>) -> BoundaryResult {
    var result: BoundaryResult;
    result.uv = uv;
    result.removed = false;
    // A NaN fails every ordered comparison, so this catches non-finite input
    // without a dedicated isNan.
    if (!(uv.x > -3.4e38 && uv.x < 3.4e38 && uv.y > -3.4e38 && uv.y < 3.4e38)) {
        result.removed = true;
        return result;
    }
    switch boundary {
        // Transparent — inclusive [0,1] acceptance, the only removing law.
        case 0u: {
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                result.removed = true;
            }
        }
        // Mirror.
        case 1u: {
            result.uv = vec2<f32>(mirror_unit(uv.x), mirror_unit(uv.y));
        }
        // Wrap.
        case 2u: {
            result.uv = vec2<f32>(wrap_unit(uv.x), wrap_unit(uv.y));
        }
        // Hold.
        default: {
            result.uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
        }
    }
    return result;
}

fn mapped_uv(uv: vec2<f32>, row_0: vec4<f32>, row_1: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        dot(row_0.xyz, vec3<f32>(uv, 1.0)),
        dot(row_1.xyz, vec3<f32>(uv, 1.0)),
    );
}

// ---------------------------------------------------------------------------
// Pass 1 — map both inputs' vectors into recipient-local space.
// Two sampled textures.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var pass1_vectors_a: texture_2d<f32>;
@group(0) @binding(1) var pass1_vectors_b: texture_2d<f32>;
@group(0) @binding(2) var pass1_nearest: sampler;
@group(0) @binding(3) var<uniform> pass1_uniforms: ColliderUniforms;

fn map_one_input(
    uv: vec2<f32>,
    transform: ColliderTransform,
    boundary: u32,
    admitted: bool,
    velocity: vec2<f32>,
) -> vec2<f32> {
    if (!admitted) {
        return vec2<f32>(COLLIDER_INVALID);
    }
    let donor_uv = mapped_uv(uv, transform.output_to_donor_row_0, transform.output_to_donor_row_1);
    let resolved = resolve_boundary(boundary, donor_uv);
    if (resolved.removed) {
        return vec2<f32>(COLLIDER_INVALID);
    }
    // Vectors map through the 2x2 linear part only. Translation moves
    // coordinates, never a velocity.
    return vec2<f32>(
        dot(transform.donor_to_recipient_row_0.xy, velocity),
        dot(transform.donor_to_recipient_row_1.xy, velocity),
    );
}

@fragment
fn fs_map(input: VertexOutput) -> @location(0) vec4<f32> {
    let boundary = pass1_uniforms.modes.y;
    let admitted_a = (pass1_uniforms.modes.z & 1u) != 0u;
    let admitted_b = (pass1_uniforms.modes.z & 2u) != 0u;

    // The coordinate lookup obeys the boundary independently per input, so one
    // input leaving its extent never silences the other.
    let uv_a = mapped_uv(
        input.uv,
        pass1_uniforms.input_a.output_to_donor_row_0,
        pass1_uniforms.input_a.output_to_donor_row_1,
    );
    let uv_b = mapped_uv(
        input.uv,
        pass1_uniforms.input_b.output_to_donor_row_0,
        pass1_uniforms.input_b.output_to_donor_row_1,
    );
    let resolved_a = resolve_boundary(boundary, uv_a);
    let resolved_b = resolve_boundary(boundary, uv_b);
    let raw_a = textureSampleLevel(pass1_vectors_a, pass1_nearest, resolved_a.uv, 0.0).xy;
    let raw_b = textureSampleLevel(pass1_vectors_b, pass1_nearest, resolved_b.uv, 0.0).xy;

    let a = map_one_input(input.uv, pass1_uniforms.input_a, boundary, admitted_a, raw_a);
    let b = map_one_input(input.uv, pass1_uniforms.input_b, boundary, admitted_b, raw_b);
    return vec4<f32>(a, b);
}

// ---------------------------------------------------------------------------
// Pass 2 — recombine, clamp, gate.
// Three sampled textures: the mapped pair plus both gates.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var pass2_pair: texture_2d<f32>;
@group(0) @binding(1) var pass2_gate_a: texture_2d<f32>;
@group(0) @binding(2) var pass2_gate_b: texture_2d<f32>;
@group(0) @binding(3) var pass2_nearest: sampler;
@group(0) @binding(4) var<uniform> pass2_uniforms: ColliderUniforms;

// The exact twin of `ColliderInputSample::validated`. Any non-finite or
// out-of-range component removes the WHOLE sample; a partially trusted reading
// would let one hostile component steer a mode that mixes both.
fn validated(velocity: vec2<f32>, gate: vec2<f32>) -> bool {
    if (!(velocity.x > -3.4e38 && velocity.x < 3.4e38)) { return false; }
    if (!(velocity.y > -3.4e38 && velocity.y < 3.4e38)) { return false; }
    if (abs(velocity.x) > MOTION_MAX_UV_PER_SECOND) { return false; }
    if (abs(velocity.y) > MOTION_MAX_UV_PER_SECOND) { return false; }
    if (!(gate.x >= 0.0 && gate.x <= 1.0)) { return false; }
    if (!(gate.y >= 0.0 && gate.y <= 1.0)) { return false; }
    return true;
}

fn collide_vectors(mode: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = a - b;
    let m = (a + b) / 2.0;
    var raw = vec2<f32>(0.0);
    switch mode {
        // Sum.
        case 0u: { raw = a + b; }
        // Difference.
        case 1u: { raw = d; }
        // Curl.
        case 2u: { raw = vec2<f32>(-d.y, d.x); }
        // Projection.
        case 3u: {
            let bb = dot(b, b);
            if (bb <= COLLIDER_EPSILON) {
                raw = vec2<f32>(0.0);
            } else {
                raw = b * (dot(a, b) / bb);
            }
        }
        // Collision boundary — remove the mean flow normal to disagreement.
        default: {
            let dd = dot(d, d);
            if (dd <= COLLIDER_EPSILON) {
                raw = m;
            } else {
                raw = m - d * (dot(m, d) / dd);
            }
        }
    }
    return clamp(
        raw,
        vec2<f32>(-MOTION_MAX_UV_PER_SECOND),
        vec2<f32>(MOTION_MAX_UV_PER_SECOND),
    );
}

struct DerivedOutput {
    @location(0) vector: vec4<f32>,
    @location(1) gate: vec4<f32>,
};

@fragment
fn fs_collide(input: VertexOutput) -> DerivedOutput {
    let mode = pass2_uniforms.modes.x;
    let boundary = pass2_uniforms.modes.y;
    let admitted_a = (pass2_uniforms.modes.z & 1u) != 0u;
    let admitted_b = (pass2_uniforms.modes.z & 2u) != 0u;

    let pair = textureSampleLevel(pass2_pair, pass2_nearest, input.uv, 0.0);

    // Each input's gate obeys the boundary independently, exactly as its vector
    // lookup did in pass 1.
    let uv_a = mapped_uv(
        input.uv,
        pass2_uniforms.input_a.output_to_donor_row_0,
        pass2_uniforms.input_a.output_to_donor_row_1,
    );
    let uv_b = mapped_uv(
        input.uv,
        pass2_uniforms.input_b.output_to_donor_row_0,
        pass2_uniforms.input_b.output_to_donor_row_1,
    );
    let resolved_a = resolve_boundary(boundary, uv_a);
    let resolved_b = resolve_boundary(boundary, uv_b);
    let gate_a = textureSampleLevel(pass2_gate_a, pass2_nearest, resolved_a.uv, 0.0).xy;
    let gate_b = textureSampleLevel(pass2_gate_b, pass2_nearest, resolved_b.uv, 0.0).xy;

    var output: DerivedOutput;
    // The exact invalid/zero sample. It never reuses the surviving input and
    // never reuses a prior derived field: both would present an observation the
    // collider did not make.
    output.vector = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    output.gate = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    if (!admitted_a || !admitted_b) { return output; }
    if (resolved_a.removed || resolved_b.removed) { return output; }
    if (!validated(pair.xy, gate_a) || !validated(pair.zw, gate_b)) { return output; }

    let derived = collide_vectors(mode, pair.xy, pair.zw);
    // Confidence and visibility are componentwise minima. The Faraday gate then
    // applies threshold/softness/occlusion exactly once, downstream, in
    // motion_apply.wgsl and motion_refresh.wgsl — never here.
    output.vector = vec4<f32>(derived, 0.0, 1.0);
    output.gate = vec4<f32>(min(gate_a.x, gate_b.x), min(gate_a.y, gate_b.y), 0.0, 1.0);
    return output;
}
