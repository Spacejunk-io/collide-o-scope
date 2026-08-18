// Dedicated Symmetry Field pass.
//
// `renderer::symmetry_field` prepends the canonical blend.wgsl source, exactly
// as the Collision Rack does, so the wet law here is the engine-wide one.
//
// This is the only shader in the project that binds eight sampled textures in
// one pass, and the only one that binds NO sampler at all. Every lookup is an
// explicit `textureLoad`, so nothing here depends on filterable formats, on
// implicit derivatives, or on uniform control flow. The four-load covered
// bilinear is `rack_node.wgsl:60 source_premultiplied_linear` transplanted onto
// each input; the D2 array form is the same filter with a clamped layer index.
//
// Eight bindings across THREE bind groups, in the order the Rust layouts
// declare them:
//   group 0 (image):  0 carrier, 1 donor 0, 2 donor 1,
//                     3 the Compat8 clean-history D2 array;
//   group 1 (uniform): the 1,024-byte dynamic-offset record;
//   group 2 (motion): 0/1 slot 0 vectors/gates, 2/3 slot 1 vectors/gates.
//
// The motion pair sits in its own group because a `MotionGpuField` owns a
// committed ping/pong parity of its own, independent of the carrier parity and
// of the N-1 tap parity. Held in one group those three dimensions would
// multiply — 4 x 4 = 16 prebuilt groups per node; split, they add: 4 image
// groups (carrier x N-1) plus 4 motion groups (the two slots' committed
// parities) = 8 per node. This is a deliberate, documented deviation from the
// frozen "Bind groups: 2" row, taken so an authored motion route can actually
// reach the pixels while warm encode still allocates nothing.
//
// Worst-case explicit texture operations per pixel is TEN: the dry carrier's
// four loads, the sector source's four loads, and one vector plus one gate
// load. The sector source is chosen BEFORE the filter runs, so a donor sector
// and a carrier sector cost the same four operations and the ledger is
// donor-state independent.
//
// Surfaces are linear-light STRAIGHT alpha RGBA16Float, except the clean
// history, which is Rgba8UnormSrgb and therefore already sRGB-decoded to linear
// light by `textureLoad` at eight bits per channel.

// Mode codes mirror `symmetry::SymmetryMode::code`. Permanent, append-only.
const SYM_CYCLIC: u32 = 0u;
const SYM_DIHEDRAL: u32 = 1u;
const SYM_PLANAR_P1: u32 = 2u;
const SYM_PLANAR_PM: u32 = 3u;
const SYM_PLANAR_P2: u32 = 4u;
const SYM_PLANAR_PMM: u32 = 5u;
const SYM_LOG_SPIRAL: u32 = 6u;
const SYM_ORBIT: u32 = 7u;

// Boundary codes mirror `symmetry::SymmetryBoundary::code`. Codes 0..3 are
// byte-compatible with the Displace boundary vocabulary; 4 is appended.
const SYM_BOUNDARY_TRANSPARENT: u32 = 0u;
const SYM_BOUNDARY_MIRROR: u32 = 1u;
const SYM_BOUNDARY_WRAP: u32 = 2u;
const SYM_BOUNDARY_HOLD: u32 = 3u;
const SYM_BOUNDARY_CELLULAR_REENTRY: u32 = 4u;

// Source codes mirror `symmetry::SymmetrySource::code`.
const SYM_SOURCE_CARRIER: u32 = 0u;
const SYM_SOURCE_DONOR0: u32 = 1u;
const SYM_SOURCE_DONOR1: u32 = 2u;
const SYM_SOURCE_CLEAN_HISTORY: u32 = 3u;

const SYM_SECTOR_RECORDS: u32 = 32u;
const SYM_TAU: f32 = 6.2831855;

// Bounded log-spiral quotient constants, mirroring the Rust module. The anchor
// log radius is computed rather than written as a literal so the two sides
// cannot drift.
const SYM_SPIRAL_MIN_LOG_PERIOD: f32 = 0.25;
const SYM_SPIRAL_ANCHOR_RADIUS: f32 = 0.25;
const SYM_SPIRAL_MIN_RADIUS: f32 = 0.0001;

// A sector's motion vector is UV per second. One reference tick — never wall
// clock — converts it into a bounded displacement, so the same authored state
// produces the same offset live and offline.
const SYM_MOTION_REFERENCE_SECONDS: f32 = 0.0333333333;

const SYM_FILTER_EPSILON: f32 = 0.000001;

struct SymmetryFieldUniforms {
    // 0: [mode, boundary, folds, rotations]
    // 1: [source mask bits, motion mask bits, seed, orbit relabel offset]
    // 2: [donor 0 bound, donor 1 bound, motion 0 bound, motion 1 bound]
    // 3: [history write index, history valid, sector records, exact bypass]
    node_meta: array<vec4u, 4>,
    // 0: [center x, center y, sector width, cell period]
    // 1: [radial phase, planar axis, planar phase, spiral step]
    // 2: [orbit radius, orbit spin, output aspect, cell skew]
    // 3: [motion gain, hue span, reserved, reserved]
    params: array<vec4f, 4>,
    // Four rows per motion slot. Row 0 is [grid w, grid h, 1/w, 1/h]; row 1 is
    // [bound, slot, reserved, reserved]; rows 2 and 3 are reserved.
    motion_rows: array<vec4f, 8>,
    // One lane per sector: [source code, motion code, history age, hue bits].
    sectors: array<vec4u, 32>,
    // Renderer-owned: [wet, program time seconds, output width, output height].
    frame: vec4f,
    // Renderer-owned: [blend code, reserved, reserved, reserved].
    frame_modes: vec4u,
    reserved: array<vec4u, 14>,
};

@group(0) @binding(0) var carrier_tex: texture_2d<f32>;
@group(0) @binding(1) var donor0_tex: texture_2d<f32>;
@group(0) @binding(2) var donor1_tex: texture_2d<f32>;
@group(0) @binding(3) var clean_history_tex: texture_2d_array<f32>;
@group(1) @binding(0) var<uniform> field: SymmetryFieldUniforms;
@group(2) @binding(0) var motion0_vector_tex: texture_2d<f32>;
@group(2) @binding(1) var motion0_gate_tex: texture_2d<f32>;
@group(2) @binding(2) var motion1_vector_tex: texture_2d<f32>;
@group(2) @binding(3) var motion1_gate_tex: texture_2d<f32>;

// ---------------------------------------------------------------------------
// Uniform accessors. Naming the lanes once keeps the geometry below readable
// and keeps every packed offset in a single place.
// ---------------------------------------------------------------------------

fn sym_mode() -> u32 { return field.node_meta[0].x; }
fn sym_boundary() -> u32 { return field.node_meta[0].y; }
fn sym_folds() -> u32 { return max(field.node_meta[0].z, 1u); }
fn sym_rotations() -> u32 { return max(field.node_meta[0].w, 1u); }
fn sym_orbit_offset() -> u32 { return field.node_meta[1].w; }
fn sym_history_write() -> f32 { return f32(field.node_meta[3].x); }
fn sym_history_valid() -> u32 { return field.node_meta[3].y; }
fn sym_exact_bypass() -> bool { return field.node_meta[3].w != 0u; }

fn sym_center() -> vec2f { return field.params[0].xy; }
fn sym_sector_width() -> f32 { return field.params[0].z; }
fn sym_cell_period() -> f32 { return field.params[0].w; }
fn sym_frame_angle() -> f32 { return field.params[1].x; }
fn sym_axis_angle() -> f32 { return field.params[1].y; }
fn sym_planar_phase() -> f32 { return field.params[1].z; }
fn sym_spiral_step() -> f32 { return field.params[1].w; }
fn sym_orbit_radius() -> f32 { return field.params[2].x; }
fn sym_orbit_spin() -> f32 { return field.params[2].y; }
fn sym_output_aspect() -> f32 { return max(field.params[2].z, SYM_FILTER_EPSILON); }
fn sym_cell_skew() -> f32 { return field.params[2].w; }
fn sym_motion_gain() -> f32 { return field.params[3].x; }
fn sym_hue_span() -> f32 { return field.params[3].y; }

fn sym_has_lattice(mode: u32) -> bool {
    return mode == SYM_PLANAR_P1 || mode == SYM_PLANAR_PM
        || mode == SYM_PLANAR_P2 || mode == SYM_PLANAR_PMM;
}

fn sym_has_reflection(mode: u32) -> bool {
    return mode == SYM_DIHEDRAL || mode == SYM_PLANAR_PM || mode == SYM_PLANAR_PMM;
}

// ---------------------------------------------------------------------------
// Scalar helpers. Each mirrors a named Rust helper exactly.
// ---------------------------------------------------------------------------

// `spatial::rotation_matrix` composed with `spatial::apply_2x2`.
fn sym_rotate(vector: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(vector.x * c - vector.y * s, vector.x * s + vector.y * c);
}

fn sym_rem_euclid(value: f32, modulus: f32) -> f32 {
    return value - floor(value / modulus) * modulus;
}

fn sym_rem_euclid_i32(value: i32, modulus: i32) -> i32 {
    return ((value % modulus) + modulus) % modulus;
}

fn sym_div_euclid_two(value: i32) -> i32 {
    return (value - sym_rem_euclid_i32(value, 2)) / 2;
}

// `symmetry::mirror_unit`: the triangle wave shared with `effects.wgsl` edge
// mode 3 and the Displace MIRROR boundary.
fn sym_mirror_unit(value: vec2f) -> vec2f {
    return vec2f(1.0) - abs(fract(value * 0.5) * 2.0 - vec2f(1.0));
}

// ---------------------------------------------------------------------------
// The bounded log-spiral quotient. `symmetry::SymmetryDomain::spiral_period`,
// `canonical_point`, `spiral_climb`, `wrap_log` and `rescale_to_log`.
// A period of exactly zero means "no quotient": the radius passes through.
// ---------------------------------------------------------------------------

fn sym_spiral_period() -> f32 {
    if sym_mode() != SYM_LOG_SPIRAL || sym_spiral_step() == 0.0 {
        return 0.0;
    }
    let period = abs(f32(sym_folds()) * sym_spiral_step());
    return select(0.0, period, period >= SYM_SPIRAL_MIN_LOG_PERIOD);
}

fn sym_wrap_log(log_radius: f32, period: f32) -> f32 {
    let anchor = log(SYM_SPIRAL_ANCHOR_RADIUS);
    return anchor + sym_rem_euclid(log_radius - anchor, period);
}

fn sym_rescale_to_log(point: vec2f, radius: f32, log_radius: f32) -> vec2f {
    return point * (exp(log_radius) / radius);
}

fn sym_canonical_point(point: vec2f) -> vec2f {
    let period = sym_spiral_period();
    if period == 0.0 {
        return point;
    }
    let radius = length(point);
    if radius <= SYM_SPIRAL_MIN_RADIUS {
        return point;
    }
    return sym_rescale_to_log(point, radius, sym_wrap_log(log(radius), period));
}

fn sym_spiral_climb(point: vec2f, steps: u32) -> vec2f {
    if steps == 0u {
        // The identity must be exactly the identity, with no round trip
        // through the logarithm.
        return point;
    }
    let period = sym_spiral_period();
    if period == 0.0 {
        return point;
    }
    let radius = length(point);
    if radius <= SYM_SPIRAL_MIN_RADIUS {
        return point;
    }
    let climbed = log(radius) + f32(steps) * sym_spiral_step();
    return sym_rescale_to_log(point, radius, sym_wrap_log(climbed, period));
}

// ---------------------------------------------------------------------------
// Coordinate frames. These are the only two places output UV enters and leaves
// the group's own coordinates, exactly as in the Rust reference.
// ---------------------------------------------------------------------------

fn sym_group_coordinates(uv: vec2f) -> vec2f {
    let offset = uv - sym_center();
    let physical = vec2f(offset.x * sym_output_aspect(), offset.y);
    if sym_has_lattice(sym_mode()) {
        let unrotated = sym_rotate(physical, -sym_axis_angle());
        let unskewed = vec2f(unrotated.x - sym_cell_skew() * unrotated.y, unrotated.y);
        // Cell 0 is centered on the authored center, and the planar phase
        // translates the primary lattice coordinate by whole cell periods.
        return vec2f(
            unskewed.x / sym_cell_period() + 0.5 + sym_planar_phase(),
            unskewed.y / sym_cell_period() + 0.5,
        );
    }
    // The radial phase conjugates the whole group frame, which is what
    // rotating the sector origin means.
    return sym_rotate(physical, -sym_frame_angle());
}

fn sym_output_coordinates(point: vec2f) -> vec2f {
    var physical: vec2f;
    if sym_has_lattice(sym_mode()) {
        let scaled = (point - vec2f(0.5)) * sym_cell_period();
        let skewed = vec2f(scaled.x + sym_cell_skew() * scaled.y, scaled.y);
        physical = sym_rotate(skewed, sym_axis_angle());
    } else {
        var framed = point;
        if sym_mode() == SYM_ORBIT {
            // The satellite frame is a presentation of Cn, deliberately applied
            // outside the group so closure is untouched.
            framed = sym_rotate(vec2f(point.x - sym_orbit_radius(), point.y), sym_orbit_spin());
        }
        physical = sym_rotate(framed, sym_frame_angle());
    }
    return vec2f(physical.x / sym_output_aspect(), physical.y) + sym_center();
}

// ---------------------------------------------------------------------------
// Classification. `local` is the representative inside the fundamental domain
// and `raw_sector` is the pre-relabel sector index.
// ---------------------------------------------------------------------------

struct SymClassification {
    local: vec2f,
    raw_sector: u32,
};

// `SymmetryDomain::apply` for the radial family, where every element carries
// the zero lattice translation.
fn sym_apply_radial(rotation: u32, reflected: bool, point: vec2f) -> vec2f {
    var mirrored = point;
    if reflected {
        mirrored = vec2f(point.x, -point.y);
    }
    let steps = rotation % sym_rotations();
    let turned = sym_rotate(mirrored, f32(steps) * sym_sector_width());
    return sym_spiral_climb(turned, steps);
}

fn sym_classify_radial(point_in: vec2f) -> SymClassification {
    let point = sym_canonical_point(point_in);
    let order = sym_rotations();
    let angle = sym_rem_euclid(atan2(point.y, point.x), SYM_TAU);
    let index = floor(angle / sym_sector_width());
    let step = u32(clamp(index, 0.0, f32(order) - 1.0));
    let within = angle - f32(step) * sym_sector_width();
    // A reflection halves the rotation step, because the far half of the step
    // folds back through the mirror at its midpoint.
    let wedge = select(sym_sector_width(), sym_sector_width() * 0.5,
        sym_has_reflection(sym_mode()));
    var rotation = step;
    var reflected = false;
    if sym_has_reflection(sym_mode()) && within > wedge {
        rotation = (step + 1u) % order;
        reflected = true;
    }
    // The inverse of a radial element: a reflected point element is its own
    // inverse, an unreflected rotation negates its step.
    var inverse_rotation = (order - rotation % order) % order;
    if reflected {
        inverse_rotation = rotation % order;
    }
    var result: SymClassification;
    result.local = sym_apply_radial(inverse_rotation, reflected, point);
    result.raw_sector = step;
    return result;
}

fn sym_mirror_fold(fraction: f32) -> vec2f {
    if fraction > 0.5 {
        return vec2f(1.0 - fraction, 1.0);
    }
    return vec2f(fraction, 0.0);
}

fn sym_classify_planar(point: vec2f) -> SymClassification {
    let cell = floor(point);
    let fraction = point - cell;
    var local = fraction;
    var mirror_u = 0;
    var mirror_v = 0;
    let mode = sym_mode();
    if mode == SYM_PLANAR_PM {
        let folded = sym_mirror_fold(fraction.x);
        local = vec2f(folded.x, fraction.y);
        mirror_u = i32(folded.y);
    } else if mode == SYM_PLANAR_P2 {
        // A two-fold rotation folds both axes together about the cell center,
        // so it has no mirror line and its wall is a seam.
        if fraction.y > 0.5 {
            local = vec2f(1.0) - fraction;
            mirror_u = 1;
            mirror_v = 1;
        }
    } else if mode == SYM_PLANAR_PMM {
        let folded_u = sym_mirror_fold(fraction.x);
        let folded_v = sym_mirror_fold(fraction.y);
        local = vec2f(folded_u.x, folded_v.x);
        mirror_u = i32(folded_u.y);
        mirror_v = i32(folded_v.y);
    }
    let lattice_x = i32(cell.x) + mirror_u;
    let lattice_y = i32(cell.y) + mirror_v;
    var result: SymClassification;
    result.local = local;
    // A planar sector varies along both lattice axes so a cell diagonal never
    // repeats a single record across a whole row.
    result.raw_sector = u32(sym_rem_euclid_i32(lattice_x + lattice_y, i32(sym_folds())));
    return result;
}

fn sym_classify(point: vec2f) -> SymClassification {
    if sym_has_lattice(sym_mode()) {
        return sym_classify_planar(point);
    }
    return sym_classify_radial(point);
}

// ---------------------------------------------------------------------------
// Boundary laws. Only Transparent removes coverage.
// ---------------------------------------------------------------------------

// `symmetry::apply_d4`, about the cell center. Bit 2 is the reflection and
// bits 0..2 are the quarter turn.
fn sym_apply_d4(element: u32, fraction: vec2f) -> vec2f {
    var point = fraction - vec2f(0.5);
    if (element & 4u) != 0u {
        point = vec2f(-point.x, point.y);
    }
    let turn = element & 3u;
    if turn == 1u {
        point = vec2f(-point.y, point.x);
    } else if turn == 2u {
        point = -point;
    } else if turn == 3u {
        point = vec2f(point.y, -point.x);
    }
    return point + vec2f(0.5);
}

// `symmetry::cellular_reentry`. One deterministic D4 cell transform: it reads
// the cell index once, selects one of eight elements, and applies it. There is
// no self-call, no loop, and no iteration count here either, so a coordinate
// arbitrarily far outside the domain costs exactly the same work.
fn sym_cellular_reentry(uv: vec2f) -> vec2f {
    let cell = floor(uv);
    let fraction = uv - cell;
    let x = i32(cell.x);
    let y = i32(cell.y);
    let parity_x = u32(sym_rem_euclid_i32(x, 2));
    let parity_y = u32(sym_rem_euclid_i32(y, 2));
    let supercell = u32(sym_rem_euclid_i32(
        sym_div_euclid_two(x) + sym_div_euclid_two(y), 2));
    return sym_apply_d4(parity_x | (parity_y << 1u) | (supercell << 2u), fraction);
}

struct SymBoundary {
    uv: vec2f,
    covered: bool,
};

fn sym_resolve_boundary(uv: vec2f) -> SymBoundary {
    var result: SymBoundary;
    result.covered = true;
    let law = sym_boundary();
    if law == SYM_BOUNDARY_MIRROR {
        result.uv = sym_mirror_unit(uv);
    } else if law == SYM_BOUNDARY_WRAP {
        result.uv = fract(uv);
    } else if law == SYM_BOUNDARY_HOLD {
        result.uv = clamp(uv, vec2f(0.0), vec2f(1.0));
    } else if law == SYM_BOUNDARY_CELLULAR_REENTRY {
        result.uv = sym_cellular_reentry(uv);
    } else {
        // Transparent keeps the speculative coordinate bounded even with no
        // coverage, exactly as `sample_source` does for edge mode 0.
        result.covered = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
        result.uv = clamp(uv, vec2f(0.0), vec2f(1.0));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Covered bilinear filters. One logical lookup is four explicit texture
// operations; the straight/premultiplied contract is `rack_node.wgsl`'s.
// ---------------------------------------------------------------------------

fn sym_source_is_bound(source: u32, age: u32) -> bool {
    if source == SYM_SOURCE_DONOR0 { return field.node_meta[2].x != 0u; }
    if source == SYM_SOURCE_DONOR1 { return field.node_meta[2].y != 0u; }
    if source == SYM_SOURCE_CLEAN_HISTORY {
        // Age 0 is the virtual current image, which for this pass is the
        // carrier itself; every stored age must be inside the materialized
        // window or the ring would answer with never-written texture memory.
        return age != 0u && age < sym_history_valid();
    }
    return true;
}

// Resolve the sector's source BEFORE any filter runs, so a bound donor, an
// unbound donor and an unmaterialized history age all cost the same four
// explicit texture operations.
fn sym_effective_source(source: u32, age: u32) -> u32 {
    return select(SYM_SOURCE_CARRIER, source, sym_source_is_bound(source, age));
}


fn sym_straight_from_premultiplied(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= SYM_FILTER_EPSILON { return vec4f(0.0); }
    return vec4f(value.rgb / alpha, alpha);
}

fn sym_cover(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    return vec4f(value.rgb * alpha, alpha);
}

struct SymFilterFrame {
    p00: vec2i,
    p10: vec2i,
    p01: vec2i,
    p11: vec2i,
    fraction: vec2f,
};

fn sym_filter_frame(uv: vec2f, dimensions: vec2i) -> SymFilterFrame {
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let maximum = dimensions - vec2i(1);
    var frame: SymFilterFrame;
    frame.p00 = clamp(base, vec2i(0), maximum);
    frame.p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    frame.p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    frame.p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    frame.fraction = fract(coordinate);
    return frame;
}

fn sym_mix_frame(c00: vec4f, c10: vec4f, c01: vec4f, c11: vec4f, fraction: vec2f) -> vec4f {
    return sym_straight_from_premultiplied(
        mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y),
    );
}

fn sym_carrier_linear(uv: vec2f) -> vec4f {
    let frame = sym_filter_frame(uv, vec2i(textureDimensions(carrier_tex)));
    return sym_mix_frame(
        sym_cover(textureLoad(carrier_tex, frame.p00, 0)),
        sym_cover(textureLoad(carrier_tex, frame.p10, 0)),
        sym_cover(textureLoad(carrier_tex, frame.p01, 0)),
        sym_cover(textureLoad(carrier_tex, frame.p11, 0)),
        frame.fraction,
    );
}

fn sym_donor0_linear(uv: vec2f) -> vec4f {
    let frame = sym_filter_frame(uv, vec2i(textureDimensions(donor0_tex)));
    return sym_mix_frame(
        sym_cover(textureLoad(donor0_tex, frame.p00, 0)),
        sym_cover(textureLoad(donor0_tex, frame.p10, 0)),
        sym_cover(textureLoad(donor0_tex, frame.p01, 0)),
        sym_cover(textureLoad(donor0_tex, frame.p11, 0)),
        frame.fraction,
    );
}

fn sym_donor1_linear(uv: vec2f) -> vec4f {
    let frame = sym_filter_frame(uv, vec2i(textureDimensions(donor1_tex)));
    return sym_mix_frame(
        sym_cover(textureLoad(donor1_tex, frame.p00, 0)),
        sym_cover(textureLoad(donor1_tex, frame.p10, 0)),
        sym_cover(textureLoad(donor1_tex, frame.p01, 0)),
        sym_cover(textureLoad(donor1_tex, frame.p11, 0)),
        frame.fraction,
    );
}

// Ring index law, transplanted from `temporal_originals.wgsl:46 wrap_layer` and
// `:203 history_age_sample`. Age 0 is the VIRTUAL current image and never
// addresses a stored layer; the caller has already substituted the carrier for
// it. The layer is additionally clamped into the real array so an unwritten or
// out-of-range layer can never be read.
fn sym_history_layer(age: u32) -> i32 {
    let history_len = f32(textureNumLayers(clean_history_tex));
    let raw = sym_rem_euclid(sym_history_write() - f32(age), history_len);
    return clamp(i32(raw), 0, i32(history_len) - 1);
}

fn sym_history_linear(uv: vec2f, age: u32) -> vec4f {
    let layer = sym_history_layer(age);
    let frame = sym_filter_frame(uv, vec2i(textureDimensions(clean_history_tex)));
    return sym_mix_frame(
        sym_cover(textureLoad(clean_history_tex, frame.p00, layer, 0)),
        sym_cover(textureLoad(clean_history_tex, frame.p10, layer, 0)),
        sym_cover(textureLoad(clean_history_tex, frame.p01, layer, 0)),
        sym_cover(textureLoad(clean_history_tex, frame.p11, layer, 0)),
        frame.fraction,
    );
}

// The exact carrier texel under this fragment. The identity and wet-zero
// branches use THIS, never the bilinear: a covered filter would divide and
// re-multiply by alpha and the default readback would stop being bit identical
// to its carrier.
fn sym_carrier_texel(position: vec2f) -> vec4f {
    let maximum = vec2i(textureDimensions(carrier_tex)) - vec2i(1);
    return textureLoad(carrier_tex, clamp(vec2i(position), vec2i(0), maximum), 0);
}

// ---------------------------------------------------------------------------
// Sector reading.
// ---------------------------------------------------------------------------

fn sym_sector_sample(source: u32, age: u32, uv: vec2f) -> vec4f {
    if source == SYM_SOURCE_DONOR0 { return sym_donor0_linear(uv); }
    if source == SYM_SOURCE_DONOR1 { return sym_donor1_linear(uv); }
    if source == SYM_SOURCE_CLEAN_HISTORY { return sym_history_linear(uv, age); }
    return sym_carrier_linear(uv);
}

// One vector plus one gate load, always, whichever slot the record names and
// whether or not it names one. The neutral views are defined zeros, so an
// unarmed sector and a lost motion donor both decode to exactly zero without
// the cost depending on the binding state.
fn sym_motion_offset(motion_code: u32, uv: vec2f) -> vec2f {
    let slot = select(0u, motion_code - 1u, motion_code != 0u);
    let row = field.motion_rows[slot * 4u];
    let grid = vec2i(max(row.xy, vec2f(1.0)));
    let texel = clamp(vec2i(floor(uv * row.xy)), vec2i(0), grid - vec2i(1));
    var vectors: vec2f;
    var gate: vec2f;
    if slot == 0u {
        vectors = textureLoad(motion0_vector_tex, texel, 0).xy;
        gate = textureLoad(motion0_gate_tex, texel, 0).xy;
    } else {
        vectors = textureLoad(motion1_vector_tex, texel, 0).xy;
        gate = textureLoad(motion1_gate_tex, texel, 0).xy;
    }
    let bound = select(field.node_meta[2].w, field.node_meta[2].z, slot == 0u);
    let armed = f32(u32(motion_code != 0u) * u32(bound != 0u));
    // `gate.x` is lattice confidence and `gate.y` is the validity/occlusion
    // lane; both must survive, exactly as `motion_apply.wgsl:47 gate_weight`
    // keeps `mix(1.0, gate.y, ...)` beside the confidence term.
    let weight = clamp(gate.x, 0.0, 1.0) * clamp(gate.y, 0.0, 1.0);
    return vectors * (weight * armed * sym_motion_gain() * SYM_MOTION_REFERENCE_SECONDS);
}

// ---------------------------------------------------------------------------
// Hue. The law is `rack_node.wgsl`'s HSL round trip, copied verbatim so the
// engine has exactly one hue rotation; a source-text test asserts the two
// bodies stay character identical.
// ---------------------------------------------------------------------------

fn rgb_to_hsl(c: vec3f) -> vec3f {
    let max_c = max(max(c.r, c.g), c.b);
    let min_c = min(min(c.r, c.g), c.b);
    let lightness = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;
    if delta < 0.001 { return vec3f(0.0, 0.0, lightness); }
    let saturation = select(
        delta / (max_c + min_c),
        delta / (2.0 - max_c - min_c),
        lightness > 0.5,
    );
    var hue: f32;
    if max_c == c.r {
        hue = (c.g - c.b) / delta + select(0.0, 6.0, c.g < c.b);
    } else if max_c == c.g {
        hue = (c.b - c.r) / delta + 2.0;
    } else {
        hue = (c.r - c.g) / delta + 4.0;
    }
    return vec3f(hue / 6.0, saturation, lightness);
}

fn hue_to_rgb(p: f32, q: f32, initial: f32) -> f32 {
    var t = initial;
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f {
    if hsl.y < 0.001 { return vec3f(hsl.z); }
    let q = select(
        hsl.z + hsl.y - hsl.z * hsl.y,
        hsl.z * (1.0 + hsl.y),
        hsl.z < 0.5,
    );
    let p = 2.0 * hsl.z - q;
    return vec3f(
        hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl.x),
        hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
    );
}

// A zero hue span is exactly no rotation, whatever the sector table says.
fn sym_hue_rotate(color: vec4f, hue_bits: u32) -> vec4f {
    let turns = bitcast<f32>(hue_bits) * sym_hue_span();
    if turns == 0.0 { return color; }
    var hsl = rgb_to_hsl(clamp(color.rgb, vec3f(0.0), vec3f(1.0)));
    hsl.x = fract(hsl.x + turns);
    return vec4f(hsl_to_rgb(hsl), color.a);
}

// ---------------------------------------------------------------------------
// Wet law. Identical in shape to `rack_node.wgsl:553 apply_node_law`.
// ---------------------------------------------------------------------------

fn sym_apply_field_law(dry: vec4f, processed: vec4f) -> vec4f {
    let wet = clamp(field.frame.x, 0.0, 1.0);
    if wet <= 0.0 { return dry; }
    var result: vec4f;
    if field.frame_modes.x == BLEND_ALPHA_CUT {
        result = vec4f(dry.rgb, dry.a * (1.0 - clamp(processed.a, 0.0, 1.0)));
    } else {
        result = vec4f(
            blend_rgb(field.frame_modes.x, clamp(dry.rgb, vec3f(0.0), vec3f(1.0)),
                clamp(processed.rgb, vec3f(0.0), vec3f(1.0))),
            clamp(processed.a, 0.0, 1.0),
        );
    }
    if wet >= 1.0 { return result; }
    let alpha = mix(clamp(dry.a, 0.0, 1.0), result.a, wet);
    let premultiplied = mix(dry.rgb * clamp(dry.a, 0.0, 1.0), result.rgb * result.a, wet);
    if alpha <= BLEND_EPSILON { return vec4f(0.0); }
    return vec4f(premultiplied / alpha, alpha);
}

struct SymmetryVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(input: SymmetryVertexOutput) -> @location(0) vec4f {
    // The exact bypass and a wet of zero take the carrier texel directly. No
    // filter, no aspect round trip, no boundary law: the result is the carrier
    // byte for byte.
    if sym_exact_bypass() || field.frame.x <= 0.0 {
        return sym_carrier_texel(input.position.xy);
    }
    let dry = sym_carrier_linear(input.uv);
    let classification = sym_classify(sym_group_coordinates(input.uv));
    let raw = sym_output_coordinates(classification.local);
    let folds = sym_folds();
    let sector = (classification.raw_sector % folds + sym_orbit_offset()) % folds;
    let record = field.sectors[sector % SYM_SECTOR_RECORDS];
    // Resolve once to obtain an in-domain coordinate for the motion field, then
    // resolve again after the offset so the authored boundary owns coverage.
    let seat = sym_resolve_boundary(raw);
    let offset = sym_motion_offset(record.y, seat.uv);
    let landing = sym_resolve_boundary(raw + offset);
    var processed = vec4f(0.0);
    if landing.covered {
        let source = sym_effective_source(record.x, record.z);
        processed = sym_hue_rotate(sym_sector_sample(source, record.z, landing.uv), record.w);
    }
    if field.frame_modes.x == 0u && field.frame.x >= 1.0 { return processed; }
    return sym_apply_field_law(dry, processed);
}
