// Small format-agnostic host operations for the advanced composition engine.
// Every sampled image is straight-alpha linear RGBA. The bus law converts to
// premultiplied form for interpolation/source-over, then returns straight RGB
// so subsequent rack and matte operations retain the engine-wide contract.

// The B8 mixing boundary rides the bus pass: wipe fields, the blend family
// at the A/B meet, the dirty-mixer fault stage, and the bus melt. All laws
// are derived from BENDR (MIT, © 2026 Steve Blythe), rewritten for this
// pass's linear-light straight/premultiplied contract; the CPU reference is
// `src/mixing_boundary.rs`, followed expression for expression. The default
// state takes the textually explicit legacy branch, so a pre-B8 frame is
// byte-identical.
struct BusUniforms {
    crossfade: f32,
    mix_mode: u32,
    wipe_invert: u32,
    wipe_rep: u32,
    wipe_soft: f32,
    wipe_x: f32,
    wipe_y: f32,
    wipe_detail: f32,
    wipe_border: f32,
    wipe_border_color: u32,
    bus_blend: u32,
    dirt: f32,
    dirt_rate: f32,
    dirt_drop: f32,
    dirt_cut: f32,
    dirt_knock: f32,
    dirt_noise: f32,
    time: f32,
    random_seed: u32,
    hist_valid: u32,
    melt: f32,
    melt_width: f32,
    melt_hold: f32,
    melt_swirl: f32,
    melt_chroma: f32,
    melt_creep: f32,
    resolution: vec2f,
    output_aspect: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// Static 8x8 Bayer order. Dither belongs only to final Advanced audience
// presentation. Compat8 temporal history/feedback uses the separately
// prepared fs_copy pipeline, so recursive memory receives ordinary format
// conversion and never accumulates this pattern. The final pattern is
// frame-invariant: a fixed input produces byte-identical presentation.
const BAYER_8X8: array<u32, 64> = array<u32, 64>(
     0u, 48u, 12u, 60u,  3u, 51u, 15u, 63u,
    32u, 16u, 44u, 28u, 35u, 19u, 47u, 31u,
     8u, 56u,  4u, 52u, 11u, 59u,  7u, 55u,
    40u, 24u, 36u, 20u, 43u, 27u, 39u, 23u,
     2u, 50u, 14u, 62u,  1u, 49u, 13u, 61u,
    34u, 18u, 46u, 30u, 33u, 17u, 45u, 29u,
    10u, 58u,  6u, 54u,  9u, 57u,  5u, 53u,
    42u, 26u, 38u, 22u, 41u, 25u, 37u, 21u,
);

@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var tex_program: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(1) @binding(0) var<uniform> bus: BusUniforms;
// The bus melt's own previous output. Bound to the 1x1 neutral when the melt
// is unarmed; `hist_valid` closes the read either way.
@group(2) @binding(0) var tex_bus_history: texture_2d<f32>;

fn premultiply(straight: vec4f) -> vec4f {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4f(straight.rgb * alpha, alpha);
}

fn premultiplied_over(top: vec4f, under: vec4f) -> vec4f {
    return top + under * (1.0 - top.a);
}

fn straight_from_premultiplied(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 {
        return vec4f(0.0);
    }
    return vec4f(value.rgb / alpha, alpha);
}

fn linear_to_srgb(value: vec3f) -> vec3f {
    let low = value * 12.92;
    let high = 1.055 * pow(value, vec3f(1.0 / 2.4)) - 0.055;
    return select(high, low, value <= vec3f(0.0031308));
}

fn srgb_to_linear(value: vec3f) -> vec3f {
    let low = value / 12.92;
    let high = pow((value + 0.055) / 1.055, vec3f(2.4));
    return select(high, low, value <= vec3f(0.04045));
}

fn ordered_compat8_dither(position: vec2f) -> f32 {
    let x = u32(position.x) & 7u;
    let y = u32(position.y) & 7u;
    let rank = BAYER_8X8[y * 8u + x];
    return (f32(rank) + 0.5) / 64.0;
}

@fragment
fn fs_copy(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(tex_a, linear_sampler, uv);
}

// This entry point is intentionally valid only for an Rgba8UnormSrgb target.
// The render target applies the final sRGB transfer, so move into encoded
// space for the one-code-value dither and back to linear for the attachment.
// Alpha is coverage, not an sRGB channel; retain it without color dither.
@fragment
fn fs_present(
    @location(0) uv: vec2f,
    @builtin(position) position: vec4f,
) -> @location(0) vec4f {
    let straight = textureSample(tex_a, linear_sampler, uv);
    let linear = clamp(straight.rgb, vec3f(0.0), vec3f(1.0));
    let encoded = linear_to_srgb(linear);
    let scaled = encoded * 255.0;
    let lower = floor(scaled);
    let fraction = fract(scaled);
    let threshold = vec3f(ordered_compat8_dither(position.xy));
    let promoted = select(vec3f(0.0), vec3f(1.0), fraction >= threshold);
    // Select the two adjacent representable codes directly. Returning their
    // exact decoded centers makes the distribution immune to small
    // implementation differences in the attachment's sRGB transfer.
    let dithered = clamp((lower + promoted) / 255.0, vec3f(0.0), vec3f(1.0));
    return vec4f(srgb_to_linear(dithered), clamp(straight.a, 0.0, 1.0));
}

// --- B8 mixing-boundary helpers. CPU reference: src/mixing_boundary.rs. ---

// The shared integer avalanche (`cellular_avalanche` in effects.wgsl).
fn mix_avalanche(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// Top 24 bits as an exact unit float, the established deterministic path.
fn mix_hash_unit(value: u32) -> f32 {
    return f32(value >> 8u) * (1.0 / 16777216.0);
}

// One keyed lane draw; avalanche(0) == 0, so a zero seed is naturally the
// unseeded stream without a branch.
fn mix_lane_unit(a: u32, b: u32, lane: u32) -> f32 {
    let mixed = a ^ (b * 0x9e3779b9u) ^ lane ^ mix_avalanche(bus.random_seed);
    return mix_hash_unit(mix_avalanche(mix_avalanche(mixed) ^ 0x68bc21ebu));
}

fn bus_back_color(code: u32) -> vec3f {
    switch code {
        case 0u: { return vec3f(1.0, 1.0, 1.0); }
        case 1u: { return vec3f(1.0, 1.0, 0.0); }
        case 2u: { return vec3f(0.0, 1.0, 1.0); }
        case 3u: { return vec3f(0.0, 1.0, 0.0); }
        case 4u: { return vec3f(1.0, 0.0, 1.0); }
        case 5u: { return vec3f(1.0, 0.0, 0.0); }
        case 6u: { return vec3f(0.0, 0.0, 1.0); }
        default: { return vec3f(0.0, 0.0, 0.0); }
    }
}

// Coherent 601 YIQ round trip — the B3 feedback-rig matrices, reused because
// the melt chroma law reconstructs RGB.
fn bus_rgb_to_yiq(rgb: vec3f) -> vec3f {
    return vec3f(
        dot(rgb, vec3f(0.299, 0.587, 0.114)),
        dot(rgb, vec3f(0.596, -0.274, -0.322)),
        dot(rgb, vec3f(0.211, -0.523, 0.312)),
    );
}

fn bus_yiq_to_rgb(yiq: vec3f) -> vec3f {
    return vec3f(
        yiq.x + 0.956 * yiq.y + 0.621 * yiq.z,
        yiq.x - 0.272 * yiq.y - 0.647 * yiq.z,
        yiq.x - 1.106 * yiq.y + 1.703 * yiq.z,
    );
}

// The analytic wipe field in [0, 1]. Only Circle is aspect-corrected — the
// shipped law — and the origin offset applies inside each MULTI tile.
fn bus_wipe_field(uv: vec2f) -> f32 {
    let rep = f32(bus.wipe_rep);
    var tu = uv;
    if bus.wipe_rep >= 2u {
        tu = fract(uv * rep);
    }
    let off = vec2f(bus.wipe_x, bus.wipe_y) * 0.5;
    let c = tu - vec2f(0.5) - off;
    let far = abs(off) + vec2f(0.5);
    let n = 2.0 + floor(bus.wipe_detail * 14.0);
    switch bus.mix_mode {
        case 1u: { return tu.x; }
        case 2u: { return 1.0 - tu.y; }
        case 3u: { return (tu.x + (1.0 - tu.y)) * 0.5; }
        case 4u: { return max(abs(c.x) / far.x, abs(c.y) / far.y); }
        case 5u: {
            let scaled = c * vec2f(bus.output_aspect, 1.0);
            let far_scaled = far * vec2f(bus.output_aspect, 1.0);
            return length(scaled) / max(length(far_scaled), 0.0001);
        }
        case 6u: { return abs(c.x) / far.x; }
        case 7u: { return abs(c.y) / far.y; }
        case 8u: { return fract(tu.x * n); }
        case 9u: { return fract(tu.y * n); }
        case 10u: { return fract(atan2(c.y, c.x) / 6.2831853 + 0.5); }
        case 11u: { return fract((tu.x + tu.y) * n * 0.5); }
        case 12u: {
            let cell_x = u32(max(floor(tu.x * n * 2.0), 0.0));
            let cell_y = u32(max(floor(tu.y * n), 0.0));
            return mix_lane_unit(cell_x, cell_y, 0x4d585701u);
        }
        default: { return 0.0; }
    }
}

// The complete mix matte at one point: field, invert, feathered threshold
// with the fader remap that keeps full A at 0 and full B at 1.
fn bus_mix_matte(uv: vec2f, fader: f32) -> f32 {
    if bus.mix_mode == 0u {
        return fader;
    }
    var d = bus_wipe_field(uv);
    if bus.wipe_invert != 0u {
        d = 1.0 - d;
    }
    let sw = max(bus.wipe_soft * 0.5, 0.002);
    let tt = fader * (1.0 + 2.0 * sw) - sw;
    return smoothstep(d - sw, d + sw, tt);
}

@fragment
fn fs_bus(@location(0) uv: vec2f) -> @location(0) vec4f {
    let fader = clamp(bus.crossfade, 0.0, 1.0);

    // --- Stage 0: the dirty-mixer event clock and the knock. ---
    var duv = uv;
    var dirt_e = 0.0;
    var dirt_tick = 0u;
    if bus.dirt > 0.002 {
        let rate = 0.5 + bus.dirt_rate * 15.0;
        let phase = max(bus.time, 0.0) * rate;
        dirt_tick = u32(floor(phase));
        let fr = fract(phase);
        if mix_lane_unit(dirt_tick, 0u, 0x4d584401u) <= bus.dirt * 0.85 {
            dirt_e = exp(-fr * (11.0 + (1.6 - 11.0) * bus.dirt));
        }
        let kn = dirt_e * bus.dirt_knock;
        if kn > 0.0005 {
            let row = u32(max(floor(uv.y * max(bus.resolution.y, 1.0)), 0.0));
            var shove = (mix_lane_unit(dirt_tick, 0u, 0x4d584402u) - 0.5) * 0.16 * kn;
            shove *= 0.4 + 0.6 * (1.0 - uv.y);
            shove += (mix_lane_unit(row, dirt_tick, 0x4d584403u) - 0.5) * 0.05 * kn;
            let hop = (mix_lane_unit(dirt_tick, 0u, 0x4d584404u) - 0.5) * 0.06 * kn;
            duv = vec2f(uv.x + shove, fract(uv.y + hop));
        }
    }

    let a = premultiply(textureSample(tex_a, linear_sampler, clamp(duv, vec2f(0.0), vec2f(1.0))));
    let program = premultiply(textureSample(tex_program, linear_sampler, uv));

    // --- Stage 1: the transition matte. Dissolve is the constant fader. ---
    var m = bus_mix_matte(duv, fader);
    var wipe_d = 0.0;
    if bus.mix_mode != 0u {
        wipe_d = bus_wipe_field(duv);
        if bus.wipe_invert != 0u {
            wipe_d = 1.0 - wipe_d;
        }
    }

    // --- Stage 2: the melt band, normal, and drag. The matte is probed at
    // four points a chosen distance out; disagreement is the edge and its
    // direction the normal. A plain dissolve has no boundary, so nothing
    // happens. ---
    var band = 0.0;
    var en = vec2f(0.0);
    if bus.melt > 0.002 {
        let r = 0.004 + bus.melt_width * 0.085;
        let rx = vec2f(r / max(bus.output_aspect, 0.0001), 0.0);
        let ry = vec2f(0.0, r);
        let m_left = bus_mix_matte(duv - rx, fader);
        let m_right = bus_mix_matte(duv + rx, fader);
        let m_down = bus_mix_matte(duv - ry, fader);
        let m_up = bus_mix_matte(duv + ry, fader);
        let low = min(min(m_left, m_right), min(m_down, m_up));
        let high = max(max(m_left, m_right), max(m_down, m_up));
        band = clamp((high - low) * 1.25, 0.0, 1.0);
        let g = vec2f(m_right - m_left, m_up - m_down);
        let g_len = length(g);
        if g_len > 0.00001 {
            en = g / g_len;
            let sa = bus.melt_swirl * 1.5707964;
            en = vec2f(cos(sa) * en.x - sin(sa) * en.y, sin(sa) * en.x + cos(sa) * en.y);
        } else {
            en = vec2f(0.0);
        }
        band *= mix(1.0, 1.0 - clamp(m, 0.0, 1.0), bus.melt_creep);
    }
    let bd = en * band * bus.melt * 0.055;
    let b = premultiply(textureSample(
        tex_b,
        linear_sampler,
        clamp(uv + bd, vec2f(0.0), vec2f(1.0)),
    ));

    // --- Stage 3: the cut — a firing can throw the crossbar to the wrong
    // input for a moment. ---
    if dirt_e > 0.001 && bus.dirt_cut > 0.002 {
        var want = 0.0;
        if mix_lane_unit(dirt_tick, 0u, 0x4d584405u) >= 0.5 {
            want = 1.0;
        }
        m = mix(m, want, clamp(dirt_e * bus.dirt_cut * 1.4, 0.0, 1.0));
    }

    // --- Stage 4: the meet. Normal is the exact legacy premultiplied
    // crossfade, kept as a textually explicit branch. ---
    var ab: vec4f;
    if bus.bus_blend == 0u {
        ab = mix(a, b, m);
    } else {
        let a_straight = straight_from_premultiplied(a);
        let b_straight = straight_from_premultiplied(b);
        let blended = blend_rgb(bus.bus_blend, a_straight.rgb, b_straight.rgb);
        // Where A is transparent the incoming picture is itself; where A
        // covers, the blend law owns the meeting colour.
        let b_effective = mix(b_straight.rgb, blended, a.a);
        let weight_a = a.a * (1.0 - m);
        let weight_b = b.a * m;
        ab = vec4f(a_straight.rgb * weight_a + b_effective * weight_b, weight_a + weight_b);
    }

    // --- Stage 5: the border rule, a coverage-honest tint on the join. ---
    if bus.mix_mode != 0u && bus.wipe_border > 0.002 {
        let sw = max(bus.wipe_soft * 0.5, 0.002);
        let tt = fader * (1.0 + 2.0 * sw) - sw;
        let bw = 0.004 + bus.wipe_border * 0.1;
        let profile = 1.0 - smoothstep(bw * 0.45, bw, abs(wipe_d - tt));
        let gate = step(0.004, fader) * step(fader, 0.996);
        let border_band = profile * gate;
        if border_band > 0.001 {
            let fill = bus_back_color(bus.wipe_border_color) * ab.a;
            let opacity = border_band * clamp(bus.wipe_border * 2.5, 0.0, 1.0);
            ab = vec4f(mix(ab.rgb, fill, opacity), ab.a);
        }
    }

    // --- Stage 6: the melt hold — the stage's own last frame dissolved back
    // in inside the band, so the smear stays put and creeps outward. The
    // history tap reads the undistorted coordinate. ---
    if bus.hist_valid != 0u && band > 0.001 && bus.melt_hold > 0.002 {
        let pd = en * (0.0015 + bus.melt * 0.04);
        // Level-0 samples: these taps sit inside per-fragment branches, so
        // they must not require implicit derivatives.
        var held = textureSampleLevel(
            tex_bus_history,
            linear_sampler,
            clamp(uv + pd, vec2f(0.0), vec2f(1.0)),
            0.0,
        );
        if bus.melt_chroma > 0.002 {
            // Colour runs further than luma, the way it does off a composite
            // edge: a second, farther tap donates only its chroma pair.
            let far = textureSampleLevel(
                tex_bus_history,
                linear_sampler,
                clamp(uv + pd * (1.0 + 3.0 * bus.melt_chroma), vec2f(0.0), vec2f(1.0)),
                0.0,
            );
            let near_yiq = bus_rgb_to_yiq(held.rgb);
            let far_yiq = bus_rgb_to_yiq(far.rgb);
            held = vec4f(
                bus_yiq_to_rgb(vec3f(
                    near_yiq.x,
                    mix(near_yiq.yz, far_yiq.yz, bus.melt_chroma),
                )),
                held.a,
            );
        }
        let cap = min(0.94 + max(bus.melt_hold - 1.0, 0.0) * 0.11, 0.995);
        ab = mix(ab, held, clamp(band * bus.melt_hold, 0.0, cap));
    }

    // --- Stage 7: what is left of a firing — dropped line bands, then the
    // switching transient. Both are coverage-honest: they tint or replace
    // covered light and never mint coverage from an empty lane. ---
    if dirt_e > 0.001 {
        if bus.dirt_drop > 0.002 {
            let y_pixels = uv.y * max(bus.resolution.y, 1.0);
            let band_height = 2.0 + 26.0 * mix_lane_unit(dirt_tick, 0u, 0x4d584406u);
            let row_band = u32(max(floor(y_pixels / band_height), 0.0));
            let probability = clamp(bus.dirt_drop * dirt_e * 1.3, 0.0, 0.95);
            if mix_lane_unit(row_band, dirt_tick, 0x4d584407u) >= 1.0 - probability {
                let to_other = mix_lane_unit(row_band, dirt_tick, 0x4d584408u) >= 0.5;
                let skew = (mix_lane_unit(row_band, dirt_tick, 0x4d584409u) - 0.5) * 0.09;
                let alt_uv = clamp(vec2f(uv.x + skew, uv.y), vec2f(0.0), vec2f(1.0));
                var alt =
                    premultiply(textureSampleLevel(tex_a, linear_sampler, alt_uv, 0.0)).rgb;
                if to_other {
                    alt =
                        premultiply(textureSampleLevel(tex_b, linear_sampler, alt_uv, 0.0)).rgb;
                }
                if mix_lane_unit(row_band, dirt_tick, 0x4d58440au) >= 0.82 {
                    // Half the time the band drops through to the other side
                    // of the crossbar; sometimes to nothing but grey hash.
                    let cell = u32(max(floor(uv.x * max(bus.resolution.x, 1.0) / 2.5), 0.0));
                    let hash_value =
                        mix_lane_unit(cell, row_band ^ (dirt_tick * 0x85ebca6bu), 0x4d58440bu);
                    alt = vec3f(hash_value * 0.5) * ab.a;
                }
                ab = vec4f(
                    mix(ab.rgb, alt, clamp(bus.dirt_drop * 1.2, 0.0, 1.0)),
                    ab.a,
                );
            }
        }
        if bus.dirt_noise > 0.002 {
            let cell_x = u32(max(floor(uv.x * max(bus.resolution.x, 1.0) / 3.0), 0.0));
            let row = u32(max(floor(uv.y * max(bus.resolution.y, 1.0)), 0.0));
            let draw =
                mix_lane_unit(cell_x ^ (row * 0x9e3779b9u), dirt_tick, 0x4d58440cu) - 0.5;
            var noisy = ab.rgb + vec3f(draw * bus.dirt_noise * dirt_e * 1.6 * ab.a);
            // Colour drops out of the switching transient: desaturate toward
            // Rec.709 luma.
            let luma = dot(noisy, vec3f(0.2126, 0.7152, 0.0722));
            noisy = mix(noisy, vec3f(luma), clamp(bus.dirt_noise * dirt_e * 0.5, 0.0, 1.0));
            ab = vec4f(max(noisy, vec3f(0.0)), ab.a);
        }
    }

    return straight_from_premultiplied(premultiplied_over(program, ab));
}
