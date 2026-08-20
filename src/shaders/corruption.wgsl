// The B6 corruption trio: Block DCT, Pixel Sort, and Filter Avalanche, one
// dedicated executor with three laws. Composed with blend.wgsl (the node law
// and the shared sRGB transfer pair). Every law runs in the encoded sRGB
// domain on the straight-alpha working values — the B8 code-byte / B5
// real-codec precedent: these are storage artefacts, so quantising or
// wrapping linear light would manufacture different ones. The CPU
// references are src/block_dct.rs, src/pixel_sort.rs, and
// src/filter_avalanche.rs, followed expression for expression; the laws are
// derived from BENDR (MIT, (c) 2026 Steve Blythe).
//
// Sampler-free: every lookup is a textureLoad. Two bound textures per pass:
// t0 is the pass's primary input, t1 the carrier (DCT/sort) or the
// avalanche's retained previous output (its cold fallback binds the
// carrier, which degrades exactly to BENDR's shipped single-frame law).

struct CorruptionUniforms {
    // x: pass mode (0 dct-coef-from-carrier, 1 dct-recon-mid,
    //    2 dct-coef-from-aux, 3 dct-recon-final, 4 pixel sort, 5 avalanche)
    // y: axis (dct: 0 x / 1 y; avalanche: 0 sub / 1 up / 2 average)
    // z: the node's frozen NodeBlend code
    // w: dct block edge in texels (4..16)
    codes: vec4u,
    // x: amount, y: quantize (dct) / threshold (sort), z: hf penalty,
    // w: chroma crush
    params0: vec4f,
    // x: renderer-owned node wet, y: avalanche span in taps, z/w spare
    params1: vec4f,
    // x: master seed, y: avalanche lane epoch, z: history_valid, w spare
    keys: vec4u,
    // x/y: output dimensions in texels
    size: vec4f,
};

@group(0) @binding(0) var primary_tex: texture_2d<f32>;
@group(0) @binding(1) var carrier_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> corr: CorruptionUniforms;

const CORR_PI: f32 = 3.14159265;
const CORR_LUMA_601: vec3f = vec3f(0.299, 0.587, 0.114);
const CORR_SORT_TAPS: u32 = 32u;
const CORR_AVALANCHE_TAPS: u32 = 32u;
// The module's two hash-lane domains ("AVL" 1/2), mirroring
// filter_avalanche.rs exactly.
const CORR_LANE_FIRE: u32 = 0x41564c01u;
const CORR_LANE_DC: u32 = 0x41564c02u;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_corruption(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    return out;
}

fn corr_clamp_coord(p: vec2i) -> vec2i {
    let limit = vec2i(i32(corr.size.x) - 1, i32(corr.size.y) - 1);
    return clamp(p, vec2i(0, 0), limit);
}

// Straight linear working value -> encoded sRGB straight value.
fn corr_encode(rgb: vec3f) -> vec3f {
    return vec3f(
        blend_linear_to_srgb(clamp(rgb.r, 0.0, 1.0)),
        blend_linear_to_srgb(clamp(rgb.g, 0.0, 1.0)),
        blend_linear_to_srgb(clamp(rgb.b, 0.0, 1.0)),
    );
}

fn corr_decode(rgb: vec3f) -> vec3f {
    return vec3f(
        blend_srgb_to_linear(clamp(rgb.r, 0.0, 1.0)),
        blend_srgb_to_linear(clamp(rgb.g, 0.0, 1.0)),
        blend_srgb_to_linear(clamp(rgb.b, 0.0, 1.0)),
    );
}

fn corr_encoded_at(p: vec2i) -> vec3f {
    return corr_encode(textureLoad(carrier_tex, corr_clamp_coord(p), 0).rgb);
}

// The shared integer avalanche (cellular_avalanche), the lane_hash law from
// mixing_boundary.rs, and the top-24-bit unit float — mirrored exactly.
fn corr_hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn corr_lane_unit(a: u32, b: u32, lane: u32, seed: u32) -> f32 {
    let mixed = a ^ (b * 0x9e3779b9u) ^ lane ^ corr_hash(seed);
    let hashed = corr_hash(corr_hash(mixed) ^ 0x68bc21ebu);
    return f32(hashed >> 8u) * (1.0 / 16777216.0);
}

// The engine-wide node wet/blend law over straight-alpha values —
// `rack_node.wgsl:apply_node_law`'s shape, composed from the one canonical
// blend kernel.
fn corr_apply_field_law(dry: vec4f, processed: vec4f) -> vec4f {
    let wet = clamp(corr.params1.x, 0.0, 1.0);
    if wet <= 0.0 { return dry; }
    var result: vec4f;
    if corr.codes.z == BLEND_ALPHA_CUT {
        result = vec4f(dry.rgb, dry.a * (1.0 - clamp(processed.a, 0.0, 1.0)));
    } else {
        result = vec4f(
            blend_rgb(corr.codes.z, clamp(dry.rgb, vec3f(0.0), vec3f(1.0)),
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

// DCT-II orthonormal scale, BENDR's spelling: sqrt(1/N) at DC, sqrt(2/N)
// otherwise, applied on analysis and synthesis.
fn corr_dct_scale(index: f32, n: f32) -> f32 {
    if index < 0.5 { return sqrt(1.0 / n); }
    return sqrt(2.0 / n);
}

// One coefficient texel: this texel's within-block index is its coefficient
// index. Forward transform over the block line, then the quantiser —
// coefficient-domain chroma crush against Rec.601 luma before the round.
@fragment
fn fs_dct_stage(in: VertexOutput) -> @location(0) vec4f {
    let p = vec2i(in.position.xy);
    let n = f32(corr.codes.w);
    let axis_is_x = corr.codes.y == 0u;
    let pos = select(f32(p.y), f32(p.x), axis_is_x);
    let base = floor(pos / n) * n;
    let k = pos - base;
    if corr.codes.x == 1u {
        // Reconstruction into the next intermediate: inverse transform over
        // this block's coefficient texels, alpha carried through.
        var s = vec3f(0.0);
        for (var u = 0u; u < 16u; u++) {
            let fu = f32(u);
            if fu >= n { break; }
            var sp = vec2i(p.x, i32(base + fu));
            if axis_is_x { sp = vec2i(i32(base + fu), p.y); }
            let co = textureLoad(primary_tex, corr_clamp_coord(sp), 0).rgb;
            s += co * corr_dct_scale(fu, n) * cos((2.0 * k + 1.0) * fu * CORR_PI / (2.0 * n));
        }
        let alpha = textureLoad(primary_tex, p, 0).a;
        return vec4f(s, alpha);
    }
    // Coefficient stage. Mode 0 reads the straight-linear carrier and
    // encodes it; mode 2 reads the already-encoded intermediate.
    var co = vec3f(0.0);
    for (var x = 0u; x < 16u; x++) {
        let fx = f32(x);
        if fx >= n { break; }
        var sp = vec2i(p.x, i32(base + fx));
        if axis_is_x { sp = vec2i(i32(base + fx), p.y); }
        var s: vec3f;
        if corr.codes.x == 0u {
            s = corr_encode(textureLoad(primary_tex, corr_clamp_coord(sp), 0).rgb);
        } else {
            s = textureLoad(primary_tex, corr_clamp_coord(sp), 0).rgb;
        }
        co += s * cos((2.0 * fx + 1.0) * k * CORR_PI / (2.0 * n));
    }
    co *= corr_dct_scale(k, n);
    // The quantiser gets coarser for higher frequencies, and chroma is
    // quantised harder than luma, as it always is.
    let step = (0.004 + corr.params0.y * 0.5) * (1.0 + k * corr.params0.z * 2.0);
    let y = dot(co, CORR_LUMA_601);
    let keep = 1.0 - corr.params0.w * 0.85;
    co = vec3f(y) + (co - vec3f(y)) * keep;
    co = floor(co / step + 0.5) * step;
    // Alpha rides the chain untransformed, BENDR's own law.
    let alpha = textureLoad(primary_tex, p, 0).a;
    return vec4f(co, alpha);
}

// The final reconstruction: inverse transform of the second-axis
// coefficients, dry/wet against the carrier in the encoded domain, then the
// node law over straight-alpha working values.
@fragment
fn fs_dct_final(in: VertexOutput) -> @location(0) vec4f {
    let p = vec2i(in.position.xy);
    let n = f32(corr.codes.w);
    let axis_is_x = corr.codes.y == 0u;
    let pos = select(f32(p.y), f32(p.x), axis_is_x);
    let base = floor(pos / n) * n;
    let k = pos - base;
    var s = vec3f(0.0);
    for (var u = 0u; u < 16u; u++) {
        let fu = f32(u);
        if fu >= n { break; }
        var sp = vec2i(p.x, i32(base + fu));
        if axis_is_x { sp = vec2i(i32(base + fu), p.y); }
        let co = textureLoad(primary_tex, corr_clamp_coord(sp), 0).rgb;
        s += co * corr_dct_scale(fu, n) * cos((2.0 * k + 1.0) * fu * CORR_PI / (2.0 * n));
    }
    let dry = textureLoad(carrier_tex, p, 0);
    let dry_encoded = corr_encode(dry.rgb);
    let mixed = mix(dry_encoded, s, clamp(corr.params0.x, 0.0, 1.0));
    let processed = vec4f(corr_decode(mixed), dry.a);
    return corr_apply_field_law(dry, processed);
}

// Pixel sort: a bright pixel searches upward (decreasing y, BENDR's +y in
// GL coordinates) through at most 32 taps stepping two rows for the end of
// its run, then takes the run-end colour mixed by the amount. Taps clamp at
// the frame edge — wrapping stretched a false streak from the seam.
@fragment
fn fs_pixel_sort(in: VertexOutput) -> @location(0) vec4f {
    let p = vec2i(in.position.xy);
    let dry = textureLoad(primary_tex, p, 0);
    let enc = corr_encode(dry.rgb);
    var out_enc = enc;
    let threshold = corr.params0.y;
    if dot(enc, CORR_LUMA_601) > threshold {
        var reach = 0;
        for (var k = 1u; k <= CORR_SORT_TAPS; k++) {
            let probe = vec2i(p.x, p.y - i32(k) * 2);
            if dot(corr_encoded_at(probe), CORR_LUMA_601) <= threshold { break; }
            reach = i32(k) * 2;
        }
        let end = corr_encoded_at(vec2i(p.x, p.y - reach));
        out_enc = mix(enc, end, clamp(corr.params0.x, 0.0, 1.0));
    }
    let processed = vec4f(corr_decode(out_enc), dry.a);
    return corr_apply_field_law(dry, processed);
}

// Filter avalanche: per-lane deterministic corruption gate, bounded
// gradient accumulation from the node's own previous output (t1; the cold
// fallback binds the carrier — exactly BENDR's shipped single-frame law),
// per-lane DC seed, and the fract wrap that makes the hard hue flips.
@fragment
fn fs_avalanche(in: VertexOutput) -> @location(0) vec4f {
    let p = vec2i(in.position.xy);
    let dry = textureLoad(primary_tex, p, 0);
    let enc = corr_encode(dry.rgb);
    var out_enc = enc;
    let amount = clamp(corr.params0.x, 0.0, 1.0);
    var lane = u32(p.y);
    if corr.codes.y != 0u { lane = u32(p.x); }
    if corr_lane_unit(CORR_LANE_FIRE, corr.keys.y, lane, corr.keys.x) < amount * 0.5 {
        var step_v = vec2f(1.0, 0.0);
        if corr.codes.y == 1u { step_v = vec2f(0.0, 1.0); }
        if corr.codes.y == 2u { step_v = vec2f(0.7071, 0.7071); }
        let span = corr.params1.y;
        var acc = vec3f(0.0);
        for (var i = 1u; i <= CORR_AVALANCHE_TAPS; i++) {
            let fi = f32(i);
            if fi > span { break; }
            let tp = vec2f(p) - step_v * fi;
            // Out-of-frame taps contribute nothing but do not stop the walk.
            if tp.x < 0.0 || tp.y < 0.0 || tp.x >= corr.size.x || tp.y >= corr.size.y {
                continue;
            }
            let a1 = corr_encode(
                textureLoad(carrier_tex, corr_clamp_coord(vec2i(tp)), 0).rgb);
            let a2 = corr_encode(
                textureLoad(carrier_tex, corr_clamp_coord(vec2i(tp - step_v)), 0).rgb);
            acc += a1 - a2;
        }
        let dc = corr_lane_unit(CORR_LANE_DC, 0u, lane, corr.keys.x) * 2.0 - 1.0;
        out_enc = fract(enc + acc * amount * 1.6 + vec3f(dc * amount * 0.25));
    }
    let processed = vec4f(corr_decode(out_enc), dry.a);
    return corr_apply_field_law(dry, processed);
}
